import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import math
import traceback
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import QMessageBox

# Importiamo la logica base geometrica esistente
from beam.mesh import BeamMeshCore

class MaterialParser:
    def __init__(self, definition):
        self.segments = []
        self.tensile_limit = 1e9 
        self.comp_limit = -1e9
        
        # Default fallback
        if not definition or isinstance(definition, str):
            self.name = str(definition)
            self.segments.append({'func': '30000e6*x', 'min': -100.0, 'max': 100.0})
            self.tensile_limit = 2.0 
            return

        try:
            self.name = definition[0]
            
            # Parsing dei segmenti
            if len(definition) > 1 and isinstance(definition[1], (list, tuple)):
                for segment in definition[1:]:
                    if len(segment) >= 3:
                        formula = segment[0]
                        start = float(segment[1])
                        end = float(segment[2])
                        self.segments.append({'func': formula, 'min': start, 'max': end})
            else:
                self.segments.append({'func': '30000e6*x', 'min': -100.0, 'max': 100.0})

            # --- RILEVAMENTO LIMITI AUTOMATICO ---
            # Scansioniamo la funzione per trovare la massima tensione positiva (ft)
            # e la minima tensione negativa (fc, che sarà un valore negativo 'alto')
            test_strains = np.linspace(-0.05, 0.05, 1000) # Range ampio di strain
            max_sig = 0.0
            min_sig = 0.0
            
            for s in test_strains:
                sig, _ = self.evaluate_raw(s)
                if sig > max_sig: max_sig = sig
                if sig < min_sig: min_sig = sig
            
            # Impostiamo i limiti. Se il materiale è "piatto", mettiamo limiti di sicurezza
            self.tensile_limit = max(0.1, max_sig)
            # Se l'utente non definisce compressione, assumiamo elastico infinito (evita crash)
            self.comp_limit = min(-0.1, min_sig) if min_sig < -0.1 else -1e9

            # Override Euristici comuni per pulizia
            if "C" in self.name or "cls" in self.name.lower():
                if self.tensile_limit > 10: self.tensile_limit = 3.0 # Clamp per calcestruzzo mal definito

        except Exception:
            self.name = "Unknown"
            self.segments.append({'func': '210000e6*x', 'min': -1.0, 'max': 1.0})
            self.tensile_limit = 400.0

    def evaluate_raw(self, strain):
        """Valuta solo la funzione nuda e cruda"""
        active_seg = None
        for seg in self.segments:
            if seg['min'] <= strain <= seg['max']:
                active_seg = seg
                break
        
        if active_seg is None:
            # Fuori range definiti: elastico residuo minimo per stabilità numerica
            return 0.0, 10.0 

        def eval_str(f_str, x_val):
            try:
                return eval(f_str, {"__builtins__": None}, {"x": x_val, "abs": abs, "math": math})
            except:
                return 0.0

        h = 1e-7
        sigma = eval_str(active_seg['func'], strain)
        sigma_h = eval_str(active_seg['func'], strain + h)
        tangent = (sigma_h - sigma) / h
        
        return float(sigma), float(tangent)

    def evaluate(self, strain):
        """
        Valuta il materiale e ritorna (sigma, E_tangente).
        Gestisce casi limite e NaN.
        """
        sig, tan = self.evaluate_raw(strain)
        
        # Clamp valori numerici folli
        if math.isnan(sig) or math.isinf(sig): sig = 0.0
        if math.isnan(tan) or math.isinf(tan): tan = 1.0
        
        # Assicura una rigidezza minima per evitare matrici singolari
        if abs(tan) < 1.0: tan = 1.0
            
        return sig, tan

    def get_tensile_limit(self):
        return self.tensile_limit

class FemWorkerDG(QThread):
    finished_computation = pyqtSignal(object, object, object, object, object, object, float) 
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(str)
    progress_percent = pyqtSignal(int)

    def __init__(self, section_data, materials_db, params):
        super().__init__()
        self.section = section_data
        self.materials_db = materials_db
        self.params = params
        self.core = BeamMeshCore()

    def run(self):
        try:
            self.progress_percent.emit(0)
            self.progress_update.emit("Generazione Mesh DG (Discontinuous)...")
            
            # 1. Mesh Generation (Nodes exploded)
            mesh_data = self.generate_dg_mesh()
            self.progress_percent.emit(15)

            # 2. Solver Loop
            self.progress_update.emit("Solver Non-Lineare DG...")
            results = self.run_solver_dg(mesh_data)
            
            self.progress_percent.emit(100)
            self.finished_computation.emit(*results)

        except Exception as e:
            traceback.print_exc()
            self.error_occurred.emit(str(e))

    def generate_dg_mesh(self):
        """
        Genera una mesh dove ogni elemento ha nodi unici.
        Crea anche la lista delle 'interfacce' (facce condivise).
        """
        L_beam = self.params['L']
        nx, ny, nz = self.params['nx'], self.params['ny'], self.params['nz']
        
        min_x, max_x, min_y, max_y = self.core._get_section_bounding_box(self.section)
        # Converti in metri
        min_x, max_x = self.core._mm_to_m(min_x), self.core._mm_to_m(max_x)
        min_y, max_y = self.core._mm_to_m(min_y), self.core._mm_to_m(max_y)
        
        Lx_bbox = max_x - min_x
        Ly_bbox = max_y - min_y
        
        dx = Lx_bbox / max(1, nx) if Lx_bbox > 1e-9 else 1.0
        dy = Ly_bbox / max(1, ny) if Ly_bbox > 1e-9 else 1.0
        dz = L_beam / max(1, nz)

        xs = np.linspace(min_x + dx/2, max_x - dx/2, nx)
        ys = np.linspace(min_y + dy/2, max_y - dy/2, ny)
        
        # Identifica voxel attivi
        active_voxels = {} # Map (ix, iy, iz) -> material_name
        
        for iy in range(ny):
            for ix in range(nx):
                xc_mm = xs[ix] * 1000.0
                yc_mm = ys[iy] * 1000.0
                is_inside, mat = self.core._is_point_in_section(xc_mm, yc_mm, self.section)
                if is_inside:
                    for iz in range(nz):
                        active_voxels[(ix, iy, iz)] = mat

        coords = []
        solid_elements = [] # list of {nodes: [8 ints], mat: str, center: (x,y,z)}
        element_map = {} # (ix, iy, iz) -> element_index
        
        node_cursor = 0

        # Offsets per i nodi di un cubo unitario (ordine standard FEM Hex8)
        for key, mat in active_voxels.items():
            ix, iy, iz = key
            
            # Coordinate bounding box dell'elemento
            x0 = min_x + ix * dx; x1 = x0 + dx
            y0 = min_y + iy * dy; y1 = y0 + dy
            z0 = iz * dz;         z1 = z0 + dz
            
            # Genera 8 nodi UNICI per questo elemento
            el_nodes_indices = []
            
            # Ordine nodi Hex8
            corners = [
                (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)
            ]
            
            for c in corners:
                coords.append(list(c))
                el_nodes_indices.append(node_cursor)
                node_cursor += 1
            
            el_idx = len(solid_elements)
            solid_elements.append({
                'nodes': el_nodes_indices, 
                'mat': mat, 
                'type': 'HEX8',
                'grid_pos': (ix, iy, iz),
                'bounds': ((x0,x1), (y0,y1), (z0,z1))
            })
            element_map[key] = el_idx

        # --- GENERAZIONE INTERFACCE (COHESIVE LINKS) ---
        interfaces = []
        
        neighbor_offsets = [
            (1, 0, 0, 0, 1),  # Neighbor at X+
            (0, 1, 0, 1, 1),  # Neighbor at Y+
            (0, 0, 1, 2, 1)   # Neighbor at Z+
        ]
        
        face_map = {
            0: {'A': [1, 2, 6, 5], 'B': [0, 3, 7, 4], 'n': [1, 0, 0]},
            1: {'A': [2, 3, 7, 6], 'B': [1, 0, 4, 5], 'n': [0, 1, 0]},
            2: {'A': [4, 5, 6, 7], 'B': [0, 1, 2, 3], 'n': [0, 0, 1]}
        }

        for key, el_idx_A in element_map.items():
            ix, iy, iz = key
            
            for dix, diy, diz, axis, sign in neighbor_offsets:
                neighbor_key = (ix + dix, iy + diy, iz + diz)
                
                if neighbor_key in element_map:
                    el_idx_B = element_map[neighbor_key]
                    
                    nodes_A = solid_elements[el_idx_A]['nodes']
                    nodes_B = solid_elements[el_idx_B]['nodes']
                    
                    indices_A = [nodes_A[i] for i in face_map[axis]['A']]
                    indices_B = [nodes_B[i] for i in face_map[axis]['B']]
                    
                    # Area interfaccia
                    if axis == 0: area = dy * dz
                    elif axis == 1: area = dx * dz
                    else: area = dx * dy
                    
                    interfaces.append({
                        'nodes_A': indices_A,
                        'nodes_B': indices_B,
                        'normal': np.array(face_map[axis]['n']),
                        'area': area,
                        'mat_A': solid_elements[el_idx_A]['mat'], # Per recuperare ft
                        'mat_B': solid_elements[el_idx_B]['mat']
                    })

        solid_node_count = len(coords)
        coords = np.array(coords)

        # --- GENERAZIONE BARRE E PENALTY LINKS ---
        bar_elements = []
        penalty_links = []
        
        # Barre Longitudinali
        temp_coords = coords.tolist()
        
        for bar in self.section.get('bars', []):
            bx = self.core._mm_to_m(bar['center'][0])
            by = self.core._mm_to_m(bar['center'][1])
            diam = self.core._mm_to_m(bar['diam'])
            area = math.pi * (diam/2)**2
            mat = bar.get('material')
            
            prev_node_idx = -1
            zs = np.linspace(0, L_beam, nz + 1)
            
            for iz in range(nz + 1):
                z = zs[iz]
                curr_node_idx = len(temp_coords)
                temp_coords.append([bx, by, z])
                
                if prev_node_idx != -1:
                    bar_elements.append({
                        'nodes': [prev_node_idx, curr_node_idx],
                        'area': area, 'mat': mat, 'type': 'TRUSS_LONG'
                    })
                prev_node_idx = curr_node_idx

        # Staffe
        passo_staffe = self.params.get('stirrup_step', 0.0)
        if passo_staffe > 0:
            num_staffe = max(1, int(math.ceil(L_beam / passo_staffe)))
            for s in self.section.get('staffe', []):
                diam = self.core._mm_to_m(s.get('diam', 8))
                area = math.pi * (diam/2)**2
                mat = s.get('material')
                pts_m = [(self.core._mm_to_m(p[0]), self.core._mm_to_m(p[1])) for p in s.get('points', [])]
                if len(pts_m) < 2: continue
                
                for i in range(num_staffe):
                    z_s = (i + 0.5) * passo_staffe
                    if z_s > L_beam: continue
                    
                    first = -1; prev = -1
                    for k, p in enumerate(pts_m):
                        curr = len(temp_coords)
                        temp_coords.append([p[0], p[1], z_s])
                        if k == 0: first = curr
                        else:
                            bar_elements.append({'nodes': [prev, curr], 'area': area, 'mat': mat, 'type': 'TRUSS_STIR'})
                        prev = curr
                    if len(pts_m) > 2 and pts_m[0] != pts_m[-1]:
                         bar_elements.append({'nodes': [prev, first], 'area': area, 'mat': mat, 'type': 'TRUSS_STIR'})

        coords = np.array(temp_coords)

        # Genera Penalty Links (Collega Barre ai Solidi)
        solid_coords = coords[:solid_node_count]
        
        for i in range(solid_node_count, len(coords)):
            bar_pt = coords[i]
            dists = np.linalg.norm(solid_coords - bar_pt, axis=1)
            nearest_idx = np.argmin(dists)
            min_dist = dists[nearest_idx]
            
            if min_dist < max(dx, dy) * 1.5:
                penalty_links.append((i, nearest_idx))

        return {
            'coords': coords,
            'solid_elems': solid_elements,
            'interfaces': interfaces,
            'bar_elems': bar_elements,
            'penalty_links': penalty_links,
            'solid_node_limit': solid_node_count
        }

    def run_solver_dg(self, mesh_data):
        coords = mesh_data['coords']
        solid_elems = mesh_data['solid_elems']
        interfaces = mesh_data['interfaces']
        bar_elems = mesh_data['bar_elems']
        penalty_links = mesh_data['penalty_links']
        
        n_dof = len(coords) * 3
        u = np.zeros(n_dof)
        
        # Pre-processing solidi
        solid_predata = self._precompute_solids(coords, solid_elems)
        
        # Stato di danneggiamento (0=intatto, 1=rotto)
        interface_damage = np.zeros(len(interfaces), dtype=int)
        bar_damage = np.zeros(len(bar_elems), dtype=int)
        
        # Parametri Solver
        steps = self.params['steps']
        iters = self.params['iters']
        tol = self.params['tol']
        
        # Carichi e Vincoli
        fixed_dofs = self._get_constraints(coords, self.params['constraints'], self.params['L'])
        free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)
        
        F_ext_base = self._get_loads(coords, n_dof)
        
        u_history = [np.zeros(n_dof)]
        stress_history = [np.zeros(len(coords))]
        
        # Parametri Coesivi
        K_interface_elastic = 1e13 # N/m^3 (Rigidezza colla perfetta)
        K_penalty_link = 1e13      # Collegamento barra-solido
        
        default_mat = MaterialParser("Default")

        print(f"--- INIZIO ANALISI DG ---")
        print(f"Elementi Solidi: {len(solid_elems)}")
        print(f"Interfacce: {len(interfaces)}")
        print(f"Barre: {len(bar_elems)}")

        for step in range(1, steps + 1):
            load_factor = step / steps
            F_target = F_ext_base * load_factor
            
            progress_val = 10 + int((step / steps) * 90)
            self.progress_percent.emit(progress_val)
            self.progress_update.emit(f"Step {step}/{steps} (Carico {load_factor*100:.0f}%)")
            
            for it in range(iters):
                rows, cols, data = [], [], []
                R_int = np.zeros(n_dof)
                
                # --- 1. SOLIDI (Standard FEM Hex8 con Materiale Non-Lineare) ---
                for el_idx, el_dat in enumerate(solid_predata):
                    mat_obj = self.materials_db.get(el_dat['mat'], default_mat)
                    dof_ind = []
                    for n in el_dat['nodes']: dof_ind.extend([3*n, 3*n+1, 3*n+2])
                    u_el = u[dof_ind]
                    
                    k_el = np.zeros((24, 24))
                    r_el = np.zeros(24)
                    
                    for gp in el_dat['gps']:
                        B, vol = gp['B'], gp['detJ']
                        eps = B @ u_el
                        
                        # Calcolo Strain Scalare (Segno Positivo = Compressione Utente, Negativo = Trazione)
                        # NOTA: Convenzione Utente: + = Compressione. 
                        # Matematica FEM standard: sum(eps) < 0 è compressione volumetrica.
                        # Quindi: if fem_vol < 0 -> User Strain > 0.
                        fem_vol_strain = np.sum(eps[:3])
                        
                        # Usiamo Von Mises per magnitudo, e segno del volumetrico
                        eps_vm = math.sqrt(0.5*((eps[0]-eps[1])**2+(eps[1]-eps[2])**2+(eps[2]-eps[0])**2)+3*(eps[3]**2+eps[4]**2+eps[5]**2))
                        
                        sign = 1.0 if fem_vol_strain < 0 else -1.0
                        strain_scalar = eps_vm * sign
                        
                        # Valutazione Legge Costitutiva
                        # Ritorna Stress (sigma) e Tangente (Et) reali dalla curva
                        sigma_val, E_tan = mat_obj.evaluate(strain_scalar)
                        
                        # --- MODIFICA CRUCIALE PER ADERENZA AL CODICE 2 ---
                        # Invece di usare D_tangente per calcolare lo stress (che accumulerebbe errori),
                        # Calcoliamo il modulo SECANTE: E_sec = sigma / strain.
                        # Usiamo E_sec per calcolare le forze interne (Residuo) -> Stress corretto.
                        # Usiamo E_tan per calcolare la rigidezza (K) -> Convergenza veloce.
                        
                        if abs(strain_scalar) < 1e-12:
                            E_sec = E_tan # Limite per strain 0
                        else:
                            E_sec = sigma_val / strain_scalar

                        # Matrice D Tangente (per Stiffness K)
                        nu = 0.2
                        f_tan = E_tan / ((1+nu)*(1-2*nu))
                        D_tan = np.zeros((6,6))
                        D_tan[:3,:3] = f_tan*(1-nu); D_tan[:3,:3] -= f_tan*nu; np.fill_diagonal(D_tan[:3,:3], f_tan*(1-nu))
                        D_tan[0,1]=D_tan[0,2]=D_tan[1,0]=D_tan[1,2]=D_tan[2,0]=D_tan[2,1] = f_tan*nu
                        sh_tan = E_tan/(2*(1+nu)); D_tan[3,3]=D_tan[4,4]=D_tan[5,5] = sh_tan

                        # Matrice D Secante (per Residuo R)
                        f_sec = E_sec / ((1+nu)*(1-2*nu))
                        D_sec = np.zeros((6,6))
                        D_sec[:3,:3] = f_sec*(1-nu); D_sec[:3,:3] -= f_sec*nu; np.fill_diagonal(D_sec[:3,:3], f_sec*(1-nu))
                        D_sec[0,1]=D_sec[0,2]=D_sec[1,0]=D_sec[1,2]=D_sec[2,0]=D_sec[2,1] = f_sec*nu
                        sh_sec = E_sec/(2*(1+nu)); D_sec[3,3]=D_sec[4,4]=D_sec[5,5] = sh_sec
                        
                        # Calcoli
                        sigma_tensor = D_sec @ eps
                        r_el += (B.T @ sigma_tensor) * vol
                        k_el += (B.T @ D_tan @ B) * vol
                        
                    # Assembly
                    for r in range(24):
                        R_int[dof_ind[r]] += r_el[r]
                        for c in range(24):
                            if abs(k_el[r,c]) > 1e-9:
                                rows.append(dof_ind[r]); cols.append(dof_ind[c]); data.append(k_el[r,c])

                # --- 2. INTERFACCE (Cohesive / Penalty) ---
                # Qui gestiamo la FRATTURA (solo Trazione)
                broken_count = 0
                for i_idx, iface in enumerate(interfaces):
                    if interface_damage[i_idx] == 1:
                        broken_count += 1
                        continue 
                    
                    nA = iface['nodes_A']; nB = iface['nodes_B']
                    normal = iface['normal']
                    area = iface['area']
                    
                    # Recupero Ft dai materiali adiacenti
                    matA = self.materials_db.get(iface['mat_A'], default_mat)
                    matB = self.materials_db.get(iface['mat_B'], default_mat)
                    ft_lim = min(matA.get_tensile_limit(), matB.get_tensile_limit())
                    
                    # Calcolo Gap
                    disp_A = np.mean([u[3*n:3*n+3] for n in nA], axis=0)
                    disp_B = np.mean([u[3*n:3*n+3] for n in nB], axis=0)
                    gap_vec = disp_B - disp_A
                    gap_n = np.dot(gap_vec, normal)
                    
                    # Trazione = Rigidezza * Gap
                    traction = K_interface_elastic * gap_n
                    
                    # --- CRITERIO DI ROTTURA ---
                    # 1. Deve essere Trazione (gap_n > 0)
                    # 2. La trazione deve superare il limite Ft
                    if gap_n > 0 and traction > ft_lim:
                        interface_damage[i_idx] = 1 
                        broken_count += 1
                        continue
                    
                    # Se non rotto (o se in Compressione), applica Penalty per tenere uniti i nodi
                    # NOTA: In compressione (gap_n < 0) questo link regge infinito (contatto perfetto)
                    k_node = (K_interface_elastic * area) / 4.0
                    
                    for k in range(4):
                        idxA = [3*nA[k], 3*nA[k]+1, 3*nA[k]+2]
                        idxB = [3*nB[k], 3*nB[k]+1, 3*nB[k]+2]
                        
                        f_vec = k_node * (u[idxB] - u[idxA]) 
                        
                        R_int[idxA] -= f_vec
                        R_int[idxB] += f_vec
                        
                        for d in range(3):
                            rows.append(idxA[d]); cols.append(idxA[d]); data.append(k_node)
                            rows.append(idxB[d]); cols.append(idxB[d]); data.append(k_node)
                            rows.append(idxA[d]); cols.append(idxB[d]); data.append(-k_node)
                            rows.append(idxB[d]); cols.append(idxA[d]); data.append(-k_node)

                # --- 3. BARRE (Truss con failure) ---
                broken_bars = 0
                for b_idx, bel in enumerate(bar_elems):
                    if bar_damage[b_idx] == 1:
                        broken_bars += 1
                        continue
                        
                    mat_obj = self.materials_db.get(bel['mat'], default_mat)
                    n1, n2 = bel['nodes']
                    idx1, idx2 = [3*n1, 3*n1+1, 3*n1+2], [3*n2, 3*n2+1, 3*n2+2]
                    p1, p2 = coords[n1], coords[n2]
                    L0 = np.linalg.norm(p2 - p1)
                    if L0 < 1e-9: continue
                    
                    vec = p2 - p1
                    dir_vec = vec / L0
                    
                    u1, u2 = u[idx1], u[idx2]
                    curr_len = np.linalg.norm((p2+u2) - (p1+u1))
                    
                    strain = (curr_len - L0) / L0
                    # Segno: Utente pos = compressione. Strain calcolato: pos = allungamento.
                    # Quindi input materiale = -strain
                    strain_input = -strain
                    
                    # Valutazione materiale
                    sigma_val, Et = mat_obj.evaluate(strain_input) 
                    
                    # Check rottura barra a Trazione (strain > 0 -> strain_input < 0)
                    ft_lim_bar = mat_obj.get_tensile_limit()
                    
                    # Se strain è positivo (trazione) e sigma supera ft
                    if strain > 0 and abs(sigma_val) > ft_lim_bar and sigma_val < 0.1: 
                        # sigma_val < 0.1 è un check se la funzione materiale è già "collassata" a 0
                        # ma qui facciamo un check esplicito sui limiti
                        pass
                        
                    # Usa logica secante anche qui per coerenza
                    if abs(strain) < 1e-12: E_sec = Et
                    else: E_sec = abs(sigma_val / strain)

                    force = E_sec * strain * bel['area'] # Forza scalare (pos = trazione)
                    f_vec = force * dir_vec
                    
                    R_int[idx1] -= f_vec
                    R_int[idx2] += f_vec
                    
                    stiff = Et * bel['area'] / L0
                    K_loc = np.outer(dir_vec, dir_vec) * stiff
                    
                    for r in range(3):
                        for c in range(3):
                            val = K_loc[r,c]
                            rows.append(idx1[r]); cols.append(idx1[c]); data.append(val)
                            rows.append(idx1[r]); cols.append(idx2[c]); data.append(-val)
                            rows.append(idx2[r]); cols.append(idx1[c]); data.append(-val)
                            rows.append(idx2[r]); cols.append(idx2[c]); data.append(val)

                # --- 4. PENALTY LINKS (Bar -> Solid) ---
                for (bn, sn) in penalty_links:
                    bi, si = [3*bn, 3*bn+1, 3*bn+2], [3*sn, 3*sn+1, 3*sn+2]
                    f_pen = K_penalty_link * (u[bi] - u[si])
                    R_int[bi] += f_pen
                    R_int[si] -= f_pen
                    for k in range(3):
                        rows.extend([bi[k], si[k], bi[k], si[k]])
                        cols.extend([bi[k], si[k], si[k], bi[k]])
                        data.extend([K_penalty_link, K_penalty_link, -K_penalty_link, -K_penalty_link])

                # SOLVE
                res = F_target - R_int
                res_norm = np.linalg.norm(res[free_dofs])
                
                print(f"  Iter {it}: Residuo {res_norm:.2e} | Cracks: {broken_count}/{len(interfaces)}")
                
                if res_norm < tol:
                    break
                
                K_global = sp.coo_matrix((data, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
                K_free = K_global[free_dofs, :][:, free_dofs]
                
                try:
                    du = spla.spsolve(K_free, res[free_dofs])
                    u[free_dofs] += du
                except:
                    print("Singolarità matrice - possibile collasso totale.")
                    break

            # Fine Step: Salva storia
            u_history.append(u.copy())
            
            # --- POST PROCESSING STRESS (SOLIDI + BARRE) ---
            node_stress = np.zeros(len(coords))
            node_counts = np.zeros(len(coords))
            
            # 1. Stress Solidi
            for el_dat in solid_predata:
                u_el = u[[i for n in el_dat['nodes'] for i in (3*n, 3*n+1, 3*n+2)]]
                vm_sum = 0
                mat_obj = self.materials_db.get(el_dat['mat'], default_mat)
                
                for gp in el_dat['gps']:
                    eps = gp['B'] @ u_el
                    eps_vol = np.sum(eps[:3])
                    eps_vm = math.sqrt(0.5*((eps[0]-eps[1])**2+(eps[1]-eps[2])**2+(eps[2]-eps[0])**2)+3*(eps[3]**2+eps[4]**2+eps[5]**2))
                    sign = 1.0 if eps_vol < 0 else -1.0
                    
                    # Recupera stress reale scalare dalla curva
                    sigma_scalar, _ = mat_obj.evaluate(eps_vm * sign)
                    vm_sum += sigma_scalar
                    
                vm_avg = vm_sum / len(el_dat['gps'])
                
                for n in el_dat['nodes']:
                    node_stress[n] += vm_avg
                    node_counts[n] += 1.0 

            # 2. Stress Barre
            for bel in bar_elems:
                mat_obj = self.materials_db.get(bel['mat'], default_mat)
                n1, n2 = bel['nodes']
                idx1, idx2 = [3*n1, 3*n1+1, 3*n1+2], [3*n2, 3*n2+1, 3*n2+2]
                p1, p2 = coords[n1], coords[n2]
                L0 = np.linalg.norm(p2 - p1)
                
                if L0 > 1e-9:
                    u1, u2 = u[idx1], u[idx2]
                    curr_len = np.linalg.norm((p2+u2) - (p1+u1))
                    strain = (curr_len - L0) / L0
                    sigma, _ = mat_obj.evaluate(-strain)
                    val = abs(sigma)
                    
                    node_stress[n1] += val; node_counts[n1] += 1.0
                    node_stress[n2] += val; node_counts[n2] += 1.0

            avg_stress = np.divide(node_stress, node_counts, where=node_counts!=0)
            stress_history.append(avg_stress)

        max_disp = np.max(np.linalg.norm(u.reshape(-1, 3), axis=1))
        max_stress = np.max(stress_history[-1]) if len(stress_history) > 0 else 0
        
        return u_history, coords, solid_elems, bar_elems, max_disp, stress_history, max_stress

    def _precompute_solids(self, coords, solid_elems):
        # Helper per calcolare matrici B una volta sola
        def get_hex8_shape(xi, eta, zeta):
            pts = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
            N = np.zeros(8); dN = np.zeros((3,8))
            for i in range(8):
                px, py, pz = pts[i]
                factor = 0.125
                N[i] = factor * (1 + px*xi) * (1 + py*eta) * (1 + pz*zeta)
                dN[0,i] = factor * px * (1 + py*eta) * (1 + pz*zeta)
                dN[1,i] = factor * py * (1 + px*xi) * (1 + pz*zeta)
                dN[2,i] = factor * pz * (1 + px*xi) * (1 + py*eta)
            return N, dN

        gauss_pts = [-0.57735, 0.57735]
        data = []
        for el in solid_elems:
            el_nodes = el['nodes']
            el_coords = coords[el_nodes]
            gps = []
            for gz in gauss_pts:
                for gy in gauss_pts:
                    for gx in gauss_pts:
                        N, local_grad = get_hex8_shape(gx, gy, gz)
                        J = local_grad @ el_coords
                        detJ = np.linalg.det(J)
                        if detJ <= 0: detJ = 1e-12
                        invJ = np.linalg.inv(J)
                        dN_dX = invJ @ local_grad
                        B = np.zeros((6, 24))
                        for i in range(8):
                            idx = 3*i
                            B[0, idx] = dN_dX[0,i]
                            B[1, idx+1] = dN_dX[1,i]
                            B[2, idx+2] = dN_dX[2,i]
                            B[3, idx] = dN_dX[1,i]; B[3, idx+1] = dN_dX[0,i]
                            B[4, idx+1] = dN_dX[2,i]; B[4, idx+2] = dN_dX[1,i]
                            B[5, idx] = dN_dX[2,i]; B[5, idx+2] = dN_dX[0,i]
                        gps.append({'B': B, 'detJ': detJ})
            data.append({'gps': gps, 'nodes': el_nodes, 'mat': el['mat']})
        return data

    def _get_constraints(self, coords, constraints, L):
        fixed = []
        tol = 1e-4
        for i, c in enumerate(coords):
            x, y, z = c
            is_fix = False
            if 'x0' in constraints and abs(x - np.min(coords[:,0])) < tol: is_fix = True
            elif 'xL' in constraints and abs(x - np.max(coords[:,0])) < tol: is_fix = True
            elif 'y0' in constraints and abs(y - np.min(coords[:,1])) < tol: is_fix = True
            elif 'yL' in constraints and abs(y - np.max(coords[:,1])) < tol: is_fix = True
            elif 'z0' in constraints and abs(z - 0.0) < tol: is_fix = True
            elif 'zL' in constraints and abs(z - L) < tol: is_fix = True
            
            if is_fix: fixed.extend([3*i, 3*i+1, 3*i+2])
        return np.unique(fixed)

    def _get_loads(self, coords, n_dof):
        F = np.zeros(n_dof)
        val = self.params['load_value']
        ldir = 0 if self.params['load_dir'] == 'x' else (1 if self.params['load_dir'] == 'y' else 2)
        locs = self.params['load_locations']
        tol = 1e-4
        L = self.params['L']
        
        target_nodes = []
        for i, c in enumerate(coords):
            match = False
            x,y,z = c
            if 'x0' in locs and abs(x - np.min(coords[:,0])) < tol: match = True
            elif 'xL' in locs and abs(x - np.max(coords[:,0])) < tol: match = True
            elif 'y0' in locs and abs(y - np.min(coords[:,1])) < tol: match = True
            elif 'yL' in locs and abs(y - np.max(coords[:,1])) < tol: match = True
            elif 'z0' in locs and abs(z - 0.0) < tol: match = True
            elif 'zL' in locs and abs(z - L) < tol: match = True
            if match: target_nodes.append(i)
            
        if target_nodes:
            f_node = val / len(target_nodes)
            for n in target_nodes:
                F[3*n + ldir] += f_node
        return F

class BeamCalcolo_dg(QObject):
    def __init__(self, parent, ui, mesh_generator):
        super().__init__(parent)
        self.ui = ui
        self.mesh_generator = mesh_generator
        self.worker = None

    def start_fem_analysis(self):
        try:
            try: L = float(self.ui.beam_lunghezza.text())
            except: L = 3.0
            
            try: nx = int(self.ui.beam_definizione_x.text())
            except: nx = 8
            try: ny = int(self.ui.beam_definizione_y.text())
            except: ny = 8
            try: nz = int(self.ui.beam_definizione_z.text())
            except: nz = 10
            
            try: stirrup_step = float(self.ui.beam_passo.text())
            except: stirrup_step = 0.0

            try: load_val = float(self.ui.beam_carico.text())
            except: load_val = -1000.0
            
            if self.ui.beam_carico_direzione_x.isChecked(): load_dir = 'x'
            elif self.ui.beam_carico_direzione_y.isChecked(): load_dir = 'y'
            elif self.ui.beam_carico_direzione_z.isChecked(): load_dir = 'z'
            
            load_locs = []
            if getattr(self.ui, 'beam_carico_x0', None) and self.ui.beam_carico_x0.isChecked(): load_locs.append('x0')
            if getattr(self.ui, 'beam_carico_xL', None) and self.ui.beam_carico_xL.isChecked(): load_locs.append('xL')
            if getattr(self.ui, 'beam_carico_y0', None) and self.ui.beam_carico_y0.isChecked(): load_locs.append('y0')
            if getattr(self.ui, 'beam_carico_yL', None) and self.ui.beam_carico_yL.isChecked(): load_locs.append('yL')
            if getattr(self.ui, 'beam_carico_z0', None) and self.ui.beam_carico_z0.isChecked(): load_locs.append('z0')
            if getattr(self.ui, 'beam_carico_zL', None) and self.ui.beam_carico_zL.isChecked(): load_locs.append('zL')

            constraints = []
            if getattr(self.ui, 'beam_vincolo_x0', None) and self.ui.beam_vincolo_x0.isChecked(): constraints.append('x0')
            if getattr(self.ui, 'beam_vincolo_xL', None) and self.ui.beam_vincolo_xL.isChecked(): constraints.append('xL')
            if getattr(self.ui, 'beam_vincolo_y0', None) and self.ui.beam_vincolo_y0.isChecked(): constraints.append('y0')
            if getattr(self.ui, 'beam_vincolo_yL', None) and self.ui.beam_vincolo_yL.isChecked(): constraints.append('yL')
            if getattr(self.ui, 'beam_vincolo_z0', None) and self.ui.beam_vincolo_z0.isChecked(): constraints.append('z0')
            if getattr(self.ui, 'beam_vincolo_zL', None) and self.ui.beam_vincolo_zL.isChecked(): constraints.append('zL')

            try: steps = int(self.ui.beam_steps.text())
            except: steps = 5
            try: iters = int(self.ui.beam_iterazioni.text())
            except: iters = 10
            try: tol = float(self.ui.beam_tolleranza.text())
            except: tol = 1e-2
            try: scale_def = float(self.ui.beam_scala_deformazione.text())
            except: scale_def = 1.0

        except Exception as e:
            QMessageBox.critical(self.ui, "Errore Input", f"Errore lettura parametri GUI: {e}")
            return

        self.mesh_generator.generate_mesh() 
        sel_idx = self.mesh_generator.selected_section_index
        if sel_idx is None: sel_idx = 0
        
        try:
            mats, objs = self.mesh_generator.beam_valori.generate_matrices(sel_idx)
            materials_db = {}
            for m_def in mats:
                parser = MaterialParser(m_def)
                materials_db[parser.name] = parser
            
            section = self.mesh_generator._build_section_from_matrices(mats, objs)

        except Exception as e:
             QMessageBox.critical(self.ui, "Errore Dati", f"Errore recupero dati sezione: {e}")
             return

        params = {
            'L': L, 'nx': nx, 'ny': ny, 'nz': nz, 'stirrup_step': stirrup_step,
            'load_value': load_val, 'load_dir': load_dir, 'load_locations': load_locs,
            'constraints': constraints,
            'steps': steps, 'iters': iters, 'tol': tol
        }

        if hasattr(self.ui, 'progressBar_beam'):
            self.ui.progressBar_beam.setValue(0)

        # Utilizziamo la classe worker FemWorkerDG modificata
        self.worker = FemWorkerDG(section, materials_db, params)
        self.worker.progress_update.connect(lambda s: print(f"[DG-FEM] {s}")) 
        self.worker.error_occurred.connect(self._on_error)
        self.worker.finished_computation.connect(self._on_success)
        
        if hasattr(self.ui, 'progressBar_beam'):
            self.worker.progress_percent.connect(self.ui.progressBar_beam.setValue)

        self.target_scale = scale_def
        self.worker.start()

    def _on_error(self, msg):
        QMessageBox.critical(self.ui, "Errore FEM", msg)
        if hasattr(self.ui, 'progressBar_beam'):
            self.ui.progressBar_beam.setValue(0)

    def _on_success(self, history, coords, solid_elems, bar_elems, max_disp, stress_history, max_stress):
        print(f"[DG-FEM] Analisi completata. Max disp: {max_disp:.4f}")
        
        gl_widget = self.mesh_generator._ensure_gl_widget_in_ui()
        if gl_widget:
            gl_widget.set_fem_results(history, coords, solid_elems, bar_elems, max_disp, stress_history, max_stress)
            gl_widget.deformation_scale = self.target_scale
            gl_widget.start_animation()
        else:
            QMessageBox.warning(self.ui, "Attenzione", "Widget 3D non trovato per visualizzare i risultati.")