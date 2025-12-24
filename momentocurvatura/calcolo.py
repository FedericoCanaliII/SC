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
# MODELLO MATERIALE (Vettorializzato)
# ======================================================================
class Materiale:
    """
    Gestisce le leggi costitutive. 
    Migliorato per accettare array NumPy per calcoli ultra-rapidi.
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

        # Definizione limiti ultimi (rottura)
        if all_strains:
            self.ult_compression = min(all_strains) # es. -0.0035
            self.ult_tension = max(all_strains)     # es. 0.01
        else:
            self.ult_compression = -1e-9
            self.ult_tension = 1e-9

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
                # Valutiamo l'espressione in ambiente sicuro
                try:
                    res = eval(intervallo['expr_code'], {'__builtins__': None, 'np': np}, {'x': x})
                    # Se il risultato è scalare (es: sigma=300), lo broadcastiamo
                    sigma_out[mask] = res
                except Exception:
                    pass

        return sigma_out

# ======================================================================
# SEZIONE RINFORZATA (Ottimizzata per Mesh Vettoriale)
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

        # 1. Aggiunta Barre
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
                
                for poly, mat in self.rinforzi:
                    if poly.contains(p):
                        found_mat = mat
                        break
                
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
# THREAD CALCOLATORE (Moment-Curvature Multi-Fiber)
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

        limits_comp = np.array([m.ult_compression for m in materials])
        limits_tens = np.array([m.ult_tension for m in materials])

        thetas = np.linspace(0, 2*math.pi, n_angoli, endpoint=False)
        results = np.zeros((n_angoli, n_step_curv, 3))
        total_iter = n_angoli * n_step_curv
        count = 0

        # ---------------- CALCOLO PARALLELO ----------------
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_idx = {}
            
            for i, theta in enumerate(thetas):
                future = executor.submit(
                    self._calcola_ramo_momento_curvatura,
                    theta, n_step_curv, N_target,
                    fib_x, fib_y, fib_A, fib_mat_idx, materials,
                    limits_comp, limits_tens
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

    # ---------------- CORE SOLVER FISICO ----------------
    def _calcola_ramo_momento_curvatura(self, theta, n_steps, N_target, 
                                        x_arr, y_arr, A_arr, mat_idx_arr, 
                                        materials, lim_comp, lim_tens):
        """
        Calcola l'intera curva M-Chi per un angolo theta fissato.
        Metodo: Controllo di Curvatura + Ricerca Equilibrio Assiale.
        INTEGRA RAFFINAMENTO ADATTIVO (Bisezione) per catturare la rottura esatta.
        """
        
        # Coordinate ruotate
        cx = math.cos(theta)
        cy = math.sin(theta)
        d_arr = x_arr * cx + y_arr * cy
        
        # Dimensioni sezione in direzione di flessione
        d_min, d_max = np.min(d_arr), np.max(d_arr)
        h_sezione = d_max - d_min if (d_max - d_min) > 1 else 100.0
        
        # --------------------------------------------------------
        # IMPLEMENTAZIONE "METODO FERRARI" (Adaptive Stepping Statico)
        # --------------------------------------------------------
        
        # 1. Determinazione Curvatura Ultima Fisica
        max_strain_capacity = max(np.max(np.abs(lim_comp)), np.max(lim_tens))
        min_neutral_axis = h_sezione * 0.05
        chi_physical_limit = max_strain_capacity / min_neutral_axis
        
        # 2. Generazione Array Curvature (Distribuzione Quadratica)
        t = np.linspace(0, 1.0, n_steps)
        chis = chi_physical_limit * (t ** 2)
        chis[0] = 0.0
        
        # --------------------------------------------------------
        
        branch_data = np.zeros((n_steps, 3)) # M, Chi, Theta
        last_eps0 = 0.0
        
        for k, chi in enumerate(chis):
            # Se siamo al primo step, inizializziamo
            if chi == 0:
                eps0 = self._solve_equilibrium(0.0, d_arr, A_arr, mat_idx_arr, materials, N_target, last_eps0)
            else:
                eps0 = self._solve_equilibrium(chi, d_arr, A_arr, mat_idx_arr, materials, N_target, last_eps0)
            
            last_eps0 = eps0 
            
            # Calcolo Deformazioni Attuali
            eps_final = eps0 - chi * d_arr
            
            # --- CHECK LIMITI ROTTURA (RAFFINAMENTO ADATTIVO) ---
            is_broken = False
            
            # Check rapido sui limiti globali (bounding box strains)
            if np.min(eps_final) < np.min(lim_comp) or np.max(eps_final) > np.max(lim_tens):
                # Check preciso materiale per materiale
                for m_id, mat in enumerate(materials):
                    mask = (mat_idx_arr == m_id)
                    if not np.any(mask): continue
                    
                    mat_eps = eps_final[mask]
                    if np.min(mat_eps) < mat.ult_compression or np.max(mat_eps) > mat.ult_tension:
                        is_broken = True
                        break
            
            if is_broken:
                if k == 0:
                    branch_data[k:] = 0
                    break
                
                # --- ALGORITMO DI RAFFINAMENTO (BISEZIONE) ---
                # Abbiamo rotto tra k-1 (sano) e k (rotto). 
                # Cerchiamo il punto limite esatto per non perdere il picco di resistenza.
                
                chi_good = chis[k-1]
                chi_bad = chi
                chi_limit = chi_good # Fallback sicuro
                
                best_eps_final = None # Per salvare lo stato delle deformazioni al limite
                
                # 10 iterazioni sono sufficienti per una precisione altissima
                for _ in range(10): 
                    chi_mid = (chi_good + chi_bad) / 2.0
                    
                    # Risolvi eq per chi_mid
                    eps0_mid = self._solve_equilibrium(chi_mid, d_arr, A_arr, mat_idx_arr, materials, N_target, last_eps0)
                    eps_final_mid = eps0_mid - chi_mid * d_arr
                    
                    # Check rottura mid
                    mid_broken = False
                    for m_id, mat in enumerate(materials):
                        mask = (mat_idx_arr == m_id)
                        if not np.any(mask): continue
                        mat_eps = eps_final_mid[mask]
                        if np.min(mat_eps) < mat.ult_compression or np.max(mat_eps) > mat.ult_tension:
                            mid_broken = True
                            break
                    
                    if mid_broken:
                        chi_bad = chi_mid
                    else:
                        chi_good = chi_mid
                        chi_limit = chi_mid
                        best_eps_final = eps_final_mid
                
                # Se per assurdo non troviamo un punto (raro), usiamo step k-1
                if best_eps_final is None:
                    chi_limit = chis[k-1]
                    eps0_prev = self._solve_equilibrium(chi_limit, d_arr, A_arr, mat_idx_arr, materials, N_target, last_eps0)
                    best_eps_final = eps0_prev - chi_limit * d_arr

                # CALCOLO MOMENTO FINALE (Punto di Rottura Esatto)
                sigmas = np.zeros_like(best_eps_final)
                for m_id, mat in enumerate(materials):
                    mask = (mat_idx_arr == m_id)
                    if np.any(mask):
                        sigmas[mask] = mat.get_sigma_vectorized(best_eps_final[mask])
                
                forces = sigmas * A_arr
                M_int = -np.sum(forces * d_arr)
                M_kNm = abs(M_int) / 1e6
                
                # Salviamo il picco nello step corrente
                branch_data[k] = [M_kNm, chi_limit * 1000.0, theta]
                
                # Riempiamo il resto dell'array con l'ultimo valore valido (plateau grafico)
                if k + 1 < n_steps:
                    branch_data[k+1:] = branch_data[k]
                
                break # Uscita dal ciclo principale (Curva finita)

            # --- CALCOLO STANDARD (SE NON ROTTO) ---
            sigmas = np.zeros_like(eps_final)
            for m_id, mat in enumerate(materials):
                mask = (mat_idx_arr == m_id)
                if np.any(mask):
                    sigmas[mask] = mat.get_sigma_vectorized(eps_final[mask])
            
            forces = sigmas * A_arr
            M_int = -np.sum(forces * d_arr)
            M_kNm = abs(M_int) / 1e6
            
            branch_data[k] = [M_kNm, chi * 1000.0, theta] 

        return branch_data

    def _solve_equilibrium(self, chi, d_arr, A_arr, mat_idx_arr, materials, N_target, guess_eps0):
        """
        Trova eps0 tale che sum(sigma(eps0 - chi*d)*A) - N_target = 0
        Usa metodo Newton-Raphson o Bisezione.
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

        # 1. Prova Newton-Raphson (veloce)
        e_curr = guess_eps0
        for _ in range(15): # Aumentato leggermente per sicurezza su step non lineari
            res = residual(e_curr)
            if abs(res) < 10.0: 
                return e_curr
            
            delta = 1e-6
            res_delta = residual(e_curr + delta)
            stiffness = (res_delta - res) / delta
            
            if abs(stiffness) < 1e-3: break 
            
            e_next = e_curr - res / stiffness
            if abs(e_next - e_curr) < 1e-7:
                return e_next
            e_curr = e_next

        # 2. Bisezione (robusta)
        low, high = -0.05, 0.05 
        f_low = residual(low)
        f_high = residual(high)
        
        if f_low * f_high > 0:
            if abs(f_low) < abs(f_high):
                high = low
                low = -0.1
            else:
                low = high
                high = 0.1
        
        for _ in range(50):
            mid = (low + high) / 2
            f_mid = residual(mid)
            
            if abs(f_mid) < 10.0 or (high - low) < 1e-7:
                return mid
            
            if f_low * f_mid < 0:
                high = mid
            else:
                low = mid
                f_low = f_mid
                
        return (low + high) / 2