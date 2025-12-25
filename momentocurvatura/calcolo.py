# calcolo.py
import math
import time
import concurrent.futures
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import translate
from PyQt5.QtCore import QThread, pyqtSignal

# ======================================================================
# MODELLO MATERIALE (Vettorializzato & Ottimizzato)
# ======================================================================
class Materiale:
    """
    Gestisce le leggi costitutive. 
    Accetta array NumPy per calcoli paralleli su tutte le fibre.
    """
    def __init__(self, matrice: List[Tuple[str, float, float]], nome: str = "") -> None:
        self.nome = nome
        self.intervalli = []
        
        all_strains = []
        for i, (expr_raw, a, b) in enumerate(matrice):
            low = float(min(a, b))
            high = float(max(a, b))
            
            # Normalizzazione espressione per Python
            expr_str = expr_raw.replace('^', '**')
            
            # Pre-compilazione per velocità
            compiled = compile(expr_str, f'<mat_{nome}_{i}>', 'eval')
            
            self.intervalli.append({
                'expr_code': compiled,
                'low': low,
                'high': high,
                'raw': expr_str
            })
            all_strains.extend([low, high])

        # Definizione limiti ultimi (rottura) per il calcolo dei ratio
        if all_strains:
            self.ult_compression = min(all_strains) # es. -0.0035
            self.ult_tension = max(all_strains)     # es. 0.01
        else:
            # Fallback per materiali fittizi/infiniti
            self.ult_compression = -1e9
            self.ult_tension = 1e9

    def get_sigma_vectorized(self, eps_array: np.ndarray) -> np.ndarray:
        """
        Calcola le tensioni per un intero array di deformazioni (eps_array).
        Molto più veloce del ciclo for element-wise.
        """
        sigma_out = np.zeros_like(eps_array)
        
        # Per ogni intervallo definito nel materiale
        for intervallo in self.intervalli:
            # Maschera booleana: true dove eps è dentro l'intervallo
            mask = (eps_array >= intervallo['low']) & (eps_array <= intervallo['high'])
            
            if np.any(mask):
                # Estraiamo i valori di x per i punti validi
                x = eps_array[mask]
                # Valutiamo l'espressione in ambiente sicuro con NumPy disponibile
                try:
                    res = eval(intervallo['expr_code'], {'__builtins__': None, 'np': np}, {'x': x})
                    sigma_out[mask] = res
                except Exception:
                    pass

        return sigma_out

# ======================================================================
# SEZIONE RINFORZATA (Gestione Geometria e Mesh)
# ======================================================================
class SezioneRinforzata:
    def __init__(self, elementi: List[Tuple], materiali_dict: Dict[str, Materiale]) -> None:
        self.geometria = []
        self.barre = []
        self.rinforzi = []
        self.materiali = materiali_dict

        # Parsing Elementi
        for elem in elementi:
            tipo = elem[0]
            if tipo == 'shape':
                _, shape_type, _, mat_name, *params = elem
                mat = self.materiali.get(mat_name)
                if shape_type == 'rect':
                    self.geometria.append((self._crea_rettangolo(*params), mat))
                elif shape_type == 'poly':
                    self.geometria.append((Polygon(params[0]), mat))
                elif shape_type == 'circle':
                    self.geometria.append((Point(params[0]).buffer(params[1], 32), mat))

            elif tipo == 'bar':
                _, _, mat_name, diam, center = elem
                mat = self.materiali.get(mat_name)
                self.barre.append({'x': center[0], 'y': center[1], 'diam': diam, 'mat': mat})

            elif tipo == 'reinf':
                _, reinf_type, _, mat_name, *params = elem
                mat = self.materiali.get(mat_name)
                if reinf_type == 'rect':
                    poly = LineString([params[0], params[1]]).buffer(params[2]/2)
                    self.rinforzi.append((poly, mat))
                elif reinf_type == 'poly':
                    poly = Polygon(params[0]).buffer(params[1]/2)
                    self.rinforzi.append((poly, mat))
                elif reinf_type == 'circular':
                    poly = Point(params[0]).buffer(params[1])
                    self.rinforzi.append((poly, mat))

        self._calcola_limiti_box()
        self.area_totale = self._calcola_area_totale()

        # Cache per i dati mesh (Arrays Numpy)
        self.mesh_x: Optional[np.ndarray] = None
        self.mesh_y: Optional[np.ndarray] = None
        self.mesh_area: Optional[np.ndarray] = None
        self.mesh_mat_indices: Optional[np.ndarray] = None
        self.materiali_list: List[Materiale] = [] 

    def _crea_rettangolo(self, p1, p2):
        return Polygon([(min(p1[0],p2[0]), min(p1[1],p2[1])), 
                        (max(p1[0],p2[0]), min(p1[1],p2[1])),
                        (max(p1[0],p2[0]), max(p1[1],p2[1])), 
                        (min(p1[0],p2[0]), max(p1[1],p2[1]))])

    def _calcola_limiti_box(self):
        coords = []
        for p, _ in self.geometria + self.rinforzi:
            coords.extend(p.exterior.coords)
        for b in self.barre:
            coords.append((b['x'], b['y']))
        
        if not coords:
            self.min_x = self.max_x = self.min_y = self.max_y = 0
            return

        pts = np.array(coords)
        self.min_x, self.min_y = np.min(pts, axis=0)
        self.max_x, self.max_y = np.max(pts, axis=0)

    def _calcola_area_totale(self):
        a = sum(p.area for p, _ in self.geometria + self.rinforzi)
        a += sum(math.pi*(b['diam']/2)**2 for b in self.barre)
        return a

    def centroide_sezione(self):
        Ax = Ay = A_tot = 0.0
        for p, _ in self.geometria + self.rinforzi:
            Ax += p.centroid.x * p.area
            Ay += p.centroid.y * p.area
            A_tot += p.area
        for b in self.barre:
            area = math.pi*(b['diam']/2)**2
            Ax += b['x'] * area
            Ay += b['y'] * area
            A_tot += area
        
        if A_tot == 0: return 0,0
        return Ax/A_tot, Ay/A_tot

    def allinea_al_centro(self):
        cx, cy = self.centroide_sezione()
        self.geometria = [(translate(p, -cx, -cy), m) for p, m in self.geometria]
        self.rinforzi = [(translate(p, -cx, -cy), m) for p, m in self.rinforzi]
        for b in self.barre:
            b['x'] -= cx
            b['y'] -= cy
        self._calcola_limiti_box()
        self.mesh_x = None

    def genera_mesh_vettoriale(self, grid_step: float):
        if self.mesh_x is not None:
            return 

        punti_x = []
        punti_y = []
        aree = []
        mat_idx = []
        
        # Identifica tutti i materiali unici usati e crea una lista
        self.materiali_list = list({m for _, m in self.geometria} | 
                                   {m for _, m in self.rinforzi} | 
                                   {b['mat'] for b in self.barre})
        if None in self.materiali_list: self.materiali_list.remove(None)
        mat_to_id = {m: i for i, m in enumerate(self.materiali_list)}

        w = self.max_x - self.min_x
        h = self.max_y - self.min_y
        nx = int(math.ceil(w / grid_step)) + 1
        ny = int(math.ceil(h / grid_step)) + 1
        
        xs = np.linspace(self.min_x, self.max_x, nx)
        ys = np.linspace(self.min_y, self.max_y, ny)
        dA = (w/max(1, nx-1)) * (h/max(1, ny-1))

        # 1. Aggiunta Barre (Priorità Alta)
        for b in self.barre:
            if b['mat'] in mat_to_id:
                punti_x.append(b['x'])
                punti_y.append(b['y'])
                aree.append(math.pi * (b['diam']/2)**2)
                mat_idx.append(mat_to_id[b['mat']])

        # 2. Aggiunta Mesh Continua
        for y in ys:
            for x in xs:
                p = Point(x,y)
                found_mat = None
                
                # Check Rinforzi
                for poly, mat in self.rinforzi:
                    if poly.contains(p):
                        found_mat = mat
                        break
                
                # Check Geometria Base
                if found_mat is None:
                    for poly, mat in self.geometria:
                        if poly.contains(p):
                            found_mat = mat
                            break
                
                if found_mat and found_mat in mat_to_id:
                    punti_x.append(x)
                    punti_y.append(y)
                    aree.append(dA)
                    mat_idx.append(mat_to_id[found_mat])

        self.mesh_x = np.array(punti_x, dtype=np.float64)
        self.mesh_y = np.array(punti_y, dtype=np.float64)
        self.mesh_area = np.array(aree, dtype=np.float64)
        self.mesh_mat_indices = np.array(mat_idx, dtype=np.int32)
        
        print(f"Mesh generata: {len(self.mesh_x)} fibre.")

# ======================================================================
# THREAD CALCOLATORE (Moment-Curvature con Strain-Ratio Check)
# ======================================================================
class MomentCurvatureCalculator(QThread):
    calculation_done = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int)

    def __init__(self, sezione: SezioneRinforzata, ui, parent=None):
        super().__init__(parent)
        self.sezione = sezione
        self.ui = ui
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        # ---------------- INPUT UTENTE ----------------
        try:
            grid_step = float(self._get_ui_val('momentocurvatura_precisione', 5.0))
            n_angoli = int(self._get_ui_val('momentocurvatura_angoli', 18))
            n_step_curv = int(self._get_ui_val('momentocurvatura_step', 50))
            
            N_input_kN = - float(self._get_ui_val('momentocurvatura_N', 0))
            N_target = -N_input_kN * 1000.0 
            
        except Exception as e:
            print(f"Errore lettura UI: {e}")
            return

        # ---------------- PREPARAZIONE ----------------
        self.sezione.allinea_al_centro()
        self.sezione.genera_mesh_vettoriale(grid_step)
        
        fib_x = self.sezione.mesh_x
        fib_y = self.sezione.mesh_y
        fib_A = self.sezione.mesh_area
        fib_mat_idx = self.sezione.mesh_mat_indices
        materials = self.sezione.materiali_list

        # Pre-calcolo vettori limiti per ogni fibra (Ottimizzazione)
        # Creiamo array lunghi quanto la mesh che contengono il limite di quel materiale in quel punto
        lim_comp_expanded = np.zeros_like(fib_x)
        lim_tens_expanded = np.zeros_like(fib_x)
        
        for m_id, mat in enumerate(materials):
            mask = (fib_mat_idx == m_id)
            if np.any(mask):
                lim_comp_expanded[mask] = mat.ult_compression
                lim_tens_expanded[mask] = mat.ult_tension

        thetas = np.linspace(0, 2*math.pi, n_angoli, endpoint=False)
        results = np.zeros((n_angoli, n_step_curv, 3))
        total_iter = n_angoli * n_step_curv
        count = 0

        # ---------------- CALCOLO PARALLELO (Multithreading sugli angoli) ----------------
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_idx = {}
            
            for i, theta in enumerate(thetas):
                future = executor.submit(
                    self._calcola_ramo_robusto,
                    theta, n_step_curv, N_target,
                    fib_x, fib_y, fib_A, fib_mat_idx, materials,
                    lim_comp_expanded, lim_tens_expanded
                )
                future_to_idx[future] = i

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                if self._stop: break
                
                try:
                    branch_res = future.result()
                    results[idx, :, :] = branch_res
                except Exception as e:
                    print(f"Errore angolo {idx}: {e}")
                    import traceback
                    traceback.print_exc()

                count += n_step_curv
                self.progress.emit(int(count/total_iter * 100))

        self.calculation_done.emit(results)

    def _get_ui_val(self, name, default):
        try:
            widget = getattr(self.ui, name)
            val = widget.text() if hasattr(widget, 'text') else str(default)
            return float(val) if val else default
        except:
            return default

    # ---------------- NUOVO CORE: STRAIN-RATIO DRIVEN ----------------
    def _calcola_ramo_robusto(self, theta, n_steps, N_target, 
                              x_arr, y_arr, A_arr, mat_idx_arr, 
                              materials, lim_comp_expanded, lim_tens_expanded):
        """
        Calcola la curva M-Chi controllando rigorosamente i limiti di deformazione.
        Adatto sia per materiali duttili che fragili.
        """
        
        # Coordinate ruotate
        cx = math.cos(theta)
        cy = math.sin(theta)
        d_arr = x_arr * cx + y_arr * cy
        
        # Stima curvatura massima fisica (molto abbondante, il limitatore ratio farà il resto)
        d_range = np.max(d_arr) - np.min(d_arr)
        max_strain = max(np.max(np.abs(lim_comp_expanded)), np.max(lim_tens_expanded))
        chi_max_theoretical = (max_strain * 3.0) / (d_range * 0.1) # 3x per sicurezza
        
        # Distribuzione quadratica delle curvature per risoluzione fine all'inizio
        t = np.linspace(0, 1.0, n_steps)
        chis_target = chi_max_theoretical * (t ** 2)
        
        branch_data = np.zeros((n_steps, 3)) # M, Chi, Theta
        last_eps0 = 0.0
        
        # Variabile per tracciare se abbiamo raggiunto la rottura
        rupture_reached = False
        
        for k, chi in enumerate(chis_target):
            if rupture_reached:
                # Se rotto, manteniamo l'ultimo valore (plateau grafico) o zero
                # Copiamo il valore precedente per continuità visiva (perfettamente plastico fittizio)
                branch_data[k] = branch_data[k-1]
                continue

            # 1. Risolvi eq. assiale per curvatura corrente
            eps0 = self._solve_equilibrium(chi, d_arr, A_arr, mat_idx_arr, materials, N_target, last_eps0)
            last_eps0 = eps0
            
            # 2. Calcola deformazioni
            strains = eps0 - chi * d_arr
            
            # 3. Calcolo "Utilization Ratio" (Rapporto Sfruttamento)
            # R = eps / lim. Se R >= 1.0, rottura.
            # Gestione segni: Compressione su Compressione, Trazione su Trazione.
            
            # Maschere
            mask_tens = strains > 0
            mask_comp = strains < 0
            
            # Init ratios a -1 (sicuro)
            ratios = np.full_like(strains, -1.0)
            
            # Calcolo Trazione (dove limite > 0 e strain > 0)
            # Evita div/0 se limite è enorme o nullo (ma gestito in Materiale)
            valid_t = mask_tens & (lim_tens_expanded > 1e-6)
            if np.any(valid_t):
                ratios[valid_t] = strains[valid_t] / lim_tens_expanded[valid_t]
                
            # Calcolo Compressione (dove limite < 0 e strain < 0)
            valid_c = mask_comp & (lim_comp_expanded < -1e-6)
            if np.any(valid_c):
                # Entrambi negativi, divisione positiva
                ratios[valid_c] = strains[valid_c] / lim_comp_expanded[valid_c]

            max_ratio = np.max(ratios)
            
            # 4. LOGICA DI CONTROLLO
            if max_ratio < 1.0:
                # --- STATO SICURO ---
                # Calcola Momento normalmente
                M_val = self._calculate_moment(strains, A_arr, d_arr, mat_idx_arr, materials)
                branch_data[k] = [M_val, chi * 1000.0, theta]
                
                prev_chi = chi
                prev_eps0 = eps0
                
            else:
                # --- ROTTURA RILEVATA ---
                # La rottura è avvenuta tra prev_chi e chi corrente.
                # DOBBIAMO TROVARE ESATTAMENTE CHI_ULT TALE CHE max_ratio == 1.0
                rupture_reached = True
                
                # Bisezione per trovare chi_limit
                chi_low = prev_chi if k > 0 else 0.0
                chi_high = chi
                eps0_search = prev_eps0 if k > 0 else 0.0
                
                final_chi = chi_low
                final_eps0 = eps0_search
                
                for _ in range(15): # 15 iterazioni bastano per altissima precisione
                    chi_mid = (chi_low + chi_high) / 2.0
                    eps0_mid = self._solve_equilibrium(chi_mid, d_arr, A_arr, mat_idx_arr, materials, N_target, eps0_search)
                    strains_mid = eps0_mid - chi_mid * d_arr
                    
                    # Ricalcolo max ratio al punto medio
                    r_mid = -1.0
                    
                    # Rapido calcolo ratio max (copia logica sopra)
                    m_tens = strains_mid > 0
                    m_comp = strains_mid < 0
                    curr_max = 0.0
                    
                    # Check veloce max (ottimizzato senza array completo se possibile, ma qui usiamo numpy fast)
                    # Trazione
                    vt = m_tens & (lim_tens_expanded > 1e-6)
                    if np.any(vt):
                        curr_max = np.max(strains_mid[vt] / lim_tens_expanded[vt])
                    
                    # Compressione
                    if curr_max < 1.0: # Solo se non abbiamo già trovato un breach
                        vc = m_comp & (lim_comp_expanded < -1e-6)
                        if np.any(vc):
                            c_ratios = strains_mid[vc] / lim_comp_expanded[vc]
                            curr_max = max(curr_max, np.max(c_ratios))
                            
                    if curr_max < 1.0:
                        # Siamo ancora sicuri, alziamo il target
                        chi_low = chi_mid
                        final_chi = chi_mid
                        final_eps0 = eps0_mid
                        eps0_search = eps0_mid
                    else:
                        # Siamo rotti, abbassiamo
                        chi_high = chi_mid
                
                # Calcoliamo il momento al punto esatto di rottura
                strains_lim = final_eps0 - final_chi * d_arr
                M_lim = self._calculate_moment(strains_lim, A_arr, d_arr, mat_idx_arr, materials)
                
                # Salviamo in k
                branch_data[k] = [M_lim, final_chi * 1000.0, theta]
                
                # Riempiamo il resto dell'array con questo valore (plateau finale)
                if k + 1 < n_steps:
                    branch_data[k+1:] = branch_data[k]
                
                break # Esce dal ciclo curvature

        return branch_data

    def _calculate_moment(self, strains, A_arr, d_arr, mat_idx_arr, materials):
        """Helper per calcolare il momento date le deformazioni"""
        sigmas = np.zeros_like(strains)
        for m_id, mat in enumerate(materials):
            mask = (mat_idx_arr == m_id)
            if np.any(mask):
                sigmas[mask] = mat.get_sigma_vectorized(strains[mask])
        
        forces = sigmas * A_arr
        M_int = -np.sum(forces * d_arr)
        return abs(M_int) / 1e6

    def _solve_equilibrium(self, chi, d_arr, A_arr, mat_idx_arr, materials, N_target, guess_eps0):
        """
        Trova eps0 tale che sum(sigma(eps0 - chi*d)*A) - N_target = 0
        Usa Newton-Raphson combinato con Bisezione Safe.
        """
        
        def residual(e0):
            strains = e0 - chi * d_arr
            N_calc = 0.0
            for m_id, mat in enumerate(materials):
                mask = (mat_idx_arr == m_id)
                if np.any(mask):
                    sigs = mat.get_sigma_vectorized(strains[mask])
                    N_calc += np.sum(sigs * A_arr[mask])
            return N_calc - N_target

        # 1. Tentativo Newton-Raphson (Veloce)
        e_curr = guess_eps0
        for _ in range(10): 
            res = residual(e_curr)
            if abs(res) < 5.0:  # Tolleranza N (Newton)
                return e_curr
            
            delta = 1e-6
            res_delta = residual(e_curr + delta)
            stiffness = (res_delta - res) / delta
            
            if abs(stiffness) < 1e-3: break # Singolarità
            
            e_next = e_curr - res / stiffness
            if abs(e_next - e_curr) < 1e-7:
                return e_next
            e_curr = e_next

        # 2. Fallback Bisezione (Robusta)
        low, high = -0.1, 0.1 # Range molto ampio
        # Ottimizzazione range dinamico basata sul guess
        if guess_eps0 != 0:
            low = guess_eps0 - 0.05
            high = guess_eps0 + 0.05
            
        f_low = residual(low)
        f_high = residual(high)
        
        # Espansione range se necessario
        if f_low * f_high > 0:
            low, high = -0.5, 0.5 # Extrema ratio
            f_low = residual(low)
            f_high = residual(high)

        for _ in range(50):
            mid = (low + high) / 2
            f_mid = residual(mid)
            
            if abs(f_mid) < 5.0 or (high - low) < 1e-7:
                return mid
            
            if f_low * f_mid < 0:
                high = mid
            else:
                low = mid
                f_low = f_mid
                
        return (low + high) / 2