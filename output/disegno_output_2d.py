from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtGui import QPainter, QColor, QFont, QPen
from PyQt5.QtCore import Qt, QPoint
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

class Domain2DWidget(QOpenGLWidget):
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        
        # Dati
        self.points_3d = None          # Nuvola di punti 3D completa
        self.slice_polygon = None      # Poligono 2D calcolato (sezione)
        self.slice_n_vals = None       # Valori N per colorazione (se serve)
        
        # Impostazioni Vista
        self.view_mode = "N_Mx"        # Default
        self.pan_2d = [0.0, 0.0]
        self.zoom_2d = 1.0
        self.last_mouse_pos = None
        
        # Range dati (per scala iniziale e griglia)
        self.data_range_x = 10.0
        self.data_range_y = 10.0
        
        # Cursore
        self.cursor_pos = QPoint(0, 0)
        self.setMouseTracking(True)
        self.font = QFont("Arial", 9)

        # Configurazioni Grafiche (Stile Esempio)
        self.axis_labels = {
            "N_Mx": ("Mx [kNm]", "N [kN]"),
            "N_My": ("My [kNm]", "N [kN]"),
            "Mx_My": ("Mx [kNm]", "My [kNm]")
        }

        self.axis_colors = {
            "N": QColor(0, 200, 255),   # Ciano
            "Mx": QColor(255, 0, 0), # Rosso chiaro
            "My": QColor(0, 255, 0)  # Verde chiaro
        }

        self.verification_point = [0.0, 0.0, 0.0]  # Mx, My, N
        self.verification_color = QColor(255, 0, 0)
        self.is_inside = False

        # Connessioni UI
        try:
            self.ui.out_Mx.textChanged.connect(self.update_verification_point)
            self.ui.out_My.textChanged.connect(self.update_verification_point)
            self.ui.out_N.textChanged.connect(self.update_verification_point)
            self.ui.btn_out_verifica.clicked.connect(self.update_verification_point)
        except Exception:
            pass

    def set_points(self, points):
        """Riceve la matrice punti 3D (n_thetas, n_steps, 3) o lista piatta."""
        if points is None or len(points) == 0:
            self.points_3d = None
            self.update()
            return

        # Assicuriamoci che sia una lista piatta Nx3 per il ConvexHull
        if len(points.shape) > 2:
            self.points_3d = points.reshape(-1, 3)
        else:
            self.points_3d = points

        # Calcolo range globali per il primo adattamento vista
        if len(self.points_3d) > 0:
            # Calcoliamo i massimi assoluti per centrare bene
            mx_max = np.max(np.abs(self.points_3d[:, 0]))
            my_max = np.max(np.abs(self.points_3d[:, 1]))
            n_max = np.max(np.abs(self.points_3d[:, 2]))
            
            self.global_ranges = {
                'Mx': mx_max if mx_max > 1 else 10.0,
                'My': my_max if my_max > 1 else 10.0,
                'N': n_max if n_max > 1 else 100.0
            }
            # Aggiorna range correnti
            self._update_ranges_for_view()
            
        # Calcola intersezione iniziale
        self._compute_plane_intersection()
        self.update()

    def set_view_mode(self, mode):
        """Cambia vista ortogonale: N_Mx, N_My, Mx_My"""
        if mode not in ["N_Mx", "N_My", "Mx_My"]: return
        self.view_mode = mode
        
        self._update_ranges_for_view()
        self.pan_2d = [0.0, 0.0]
        self.zoom_2d = 1.0
        
        self._compute_plane_intersection()
        self.update()

    def _update_ranges_for_view(self):
        """Imposta data_range_x/y in base alla vista corrente"""
        if not hasattr(self, 'global_ranges'): return
        
        if self.view_mode == "N_Mx":
            self.data_range_x = self.global_ranges['Mx'] * 1.2
            self.data_range_y = self.global_ranges['N'] * 1.2
        elif self.view_mode == "N_My":
            self.data_range_x = self.global_ranges['My'] * 1.2
            self.data_range_y = self.global_ranges['N'] * 1.2
        elif self.view_mode == "Mx_My":
            self.data_range_x = self.global_ranges['Mx'] * 1.2
            self.data_range_y = self.global_ranges['My'] * 1.2

    def update_verification_point(self):
        """Legge UI, aggiorna punto verifica e ricalcola slice se cambia il piano"""
        try:
            def p(t): return float(t.replace(',', '.') or 0)
            mx = p(self.ui.out_Mx.text())
            my = p(self.ui.out_My.text())
            n  = p(self.ui.out_N.text())
            
            old = self.verification_point
            self.verification_point = [mx, my, n]
            
            # Controlla se è cambiata la coordinata che definisce il piano di taglio
            # N_Mx -> Taglio su My
            # N_My -> Taglio su Mx
            # Mx_My -> Taglio su N
            recalc = False
            if self.view_mode == "N_Mx" and abs(my - old[1]) > 1e-3: recalc = True
            elif self.view_mode == "N_My" and abs(mx - old[0]) > 1e-3: recalc = True
            elif self.view_mode == "Mx_My" and abs(n - old[2]) > 1e-3: recalc = True
            
            if recalc or self.slice_polygon is None:
                self._compute_plane_intersection()
                
            self._check_inside()
            self.update()
        except: pass

    def _compute_plane_intersection(self):
        """
        Taglia il ConvexHull 3D con il piano definito dal punto di verifica.
        Genera self.slice_polygon (Nx2)
        """
        self.slice_polygon = None
        if self.points_3d is None or len(self.points_3d) < 4: return

        # Indici assi: 0=Mx, 1=My, 2=N
        if self.view_mode == "N_Mx":   ix, iy, icut = 0, 2, 1
        elif self.view_mode == "N_My": ix, iy, icut = 1, 2, 0
        else:                          ix, iy, icut = 0, 1, 2 # Mx_My
        
        cut_val = self.verification_point[icut]
        
        try:
            hull = ConvexHull(self.points_3d)
            segments = []
            
            # Itera facce del convex hull
            for simplex in hull.simplices:
                pts = self.points_3d[simplex] # (3, 3)
                dists = pts[:, icut] - cut_val
                
                # Se i segni sono misti, c'è intersezione
                if not (np.all(dists > 0) or np.all(dists < 0)):
                    inters = []
                    # Itera lati del triangolo
                    for i in range(3):
                        j = (i + 1) % 3
                        if (dists[i] > 0 and dists[j] < 0) or (dists[i] < 0 and dists[j] > 0):
                            t = dists[i] / (dists[i] - dists[j])
                            # Interpola coordinate X e Y vista
                            px = pts[i, ix] + t * (pts[j, ix] - pts[i, ix])
                            py = pts[i, iy] + t * (pts[j, iy] - pts[i, iy])
                            inters.append([px, py])
                        elif dists[i] == 0:
                            inters.append([pts[i, ix], pts[i, iy]])
                    
                    # Rimuovi duplicati vicini
                    unique_int = []
                    for p in inters:
                        if not any(np.linalg.norm(np.array(p)-np.array(u)) < 1e-6 for u in unique_int):
                            unique_int.append(p)
                            
                    if len(unique_int) == 2:
                        segments.append(unique_int)
            
            if not segments: return

            # Unisci segmenti in un poligono ordinato (ConvexHull 2D)
            all_pts = np.array([p for s in segments for p in s])
            if len(all_pts) > 2:
                hull2 = ConvexHull(all_pts)
                self.slice_polygon = all_pts[hull2.vertices]
                
            self._check_inside()
                
        except Exception:
            self.slice_polygon = None

    def _check_inside(self):
        """Controlla se il punto è dentro la slice attuale"""
        if self.slice_polygon is None:
            self.is_inside = False
            self.verification_color = QColor(255, 0, 0)
            return

        # Coordinate punto vista
        if self.view_mode == "N_Mx":   px, py = self.verification_point[0], self.verification_point[2]
        elif self.view_mode == "N_My": px, py = self.verification_point[1], self.verification_point[2]
        else:                          px, py = self.verification_point[0], self.verification_point[1]
        
        try:
            tri = Delaunay(self.slice_polygon)
            if tri.find_simplex([px, py]) >= 0:
                self.is_inside = True
                self.verification_color = QColor(0, 255, 0) # Verde
                self.ui.out_testo_punto.setText("Verifica soddisfatta")
            else:
                self.is_inside = False
                self.verification_color = QColor(255, 0, 0) # Rosso
                self.ui.out_testo_punto.setText("Verifica non soddisfatta")
        except:
            pass

    # --- RENDERING ---

    def initializeGL(self):
        glClearColor(40/255, 40/255, 40/255, 1.0) # Sfondo Grigio Scuro (Esempio)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        
        # Setup Proiezione Ortogonale
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        glOrtho(
            -self.data_range_x * self.zoom_2d + self.pan_2d[0],
             self.data_range_x * self.zoom_2d + self.pan_2d[0],
            -self.data_range_y * self.zoom_2d + self.pan_2d[1],
             self.data_range_y * self.zoom_2d + self.pan_2d[1],
            -1.0, 1.0
        )
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()

        # 1. Griglia (Sfondo)
        self._draw_grid()
        
        # 2. Dominio (Con trasparenza per vedere la griglia sotto)
        self._draw_domain()
        
        # 3. Assi (Sopra il dominio per chiarezza, o sotto se preferisci)
        # L'esempio li disegna sopra la griglia. 
        # Disegniamo qui solo il riempimento dominio trasparente, poi gli assi?
        # L'ordine migliore per visibilità tecnica: Griglia -> Assi -> Dominio(Trasp) -> Punto -> Testi
        
        # 4. Punto Verifica
        self._draw_verification_point_gl()
        
        # 5. Overlay Testi e Assi Schermo (QPainter)
        self._draw_screen_axes_and_labels()
        self._draw_tracker_with_coords()

    def _draw_domain(self):
        """Disegna il poligono di sezione con trasparenza forzata"""
        if self.slice_polygon is None: return
        
        # --- 1. RIEMPIMENTO TRASPARENTE ---
        # Abilita il blending SOLO per questa fase
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Colore Grigio Chiaro con Alpha 0.3 (30% visibile, 70% trasparente)
        # Se lo vedi ancora troppo pieno, abbassa l'ultimo valore (es. 0.15)
        glColor4f(1, 1, 1, 0.1) 
        
        glBegin(GL_POLYGON)
        for p in self.slice_polygon:
            glVertex2f(p[0], p[1])
        glEnd()
        
        # Disabilita il blending per il resto (linee, bordi) che devono essere netti
        glDisable(GL_BLEND)
        
        # --- 2. BORDO BIANCO SOLIDO ---
        glLineWidth(1.0)
        glColor4f(1.0, 1.0, 1.0, 1.0) # Alpha 1.0 = Opaco
        glBegin(GL_LINE_LOOP)
        for p in self.slice_polygon:
            glVertex2f(p[0], p[1])
        glEnd()

    def _draw_verification_point_gl(self):
        """Disegna punto GL"""
        if self.view_mode == "N_Mx":   x, y = self.verification_point[0], self.verification_point[2]
        elif self.view_mode == "N_My": x, y = self.verification_point[1], self.verification_point[2]
        else:                          x, y = self.verification_point[0], self.verification_point[1]
        
        # Quadrato colorato
        col = self.verification_color
        glColor3f(col.redF(), col.greenF(), col.blueF())
        
        glPointSize(10.0)
        glBegin(GL_POINTS)
        glVertex2f(x, y)
        glEnd()

    def _draw_grid(self):
        # Limiti vista world
        wx_min = -self.data_range_x * self.zoom_2d + self.pan_2d[0]
        wx_max =  self.data_range_x * self.zoom_2d + self.pan_2d[0]
        wy_min = -self.data_range_y * self.zoom_2d + self.pan_2d[1]
        wy_max =  self.data_range_y * self.zoom_2d + self.pan_2d[1]

        tx = self._tick(wx_max - wx_min)
        ty = self._tick(wy_max - wy_min)

        glColor3f(0.2, 0.2, 0.2)
        glLineWidth(1)

        glBegin(GL_LINES)
        # Verticali
        x = np.floor(wx_min / tx) * tx
        while x <= wx_max + 1e-12:
            glVertex2f(x, wy_min); glVertex2f(x, wy_max)
            x += tx
        # Orizzontali
        y = np.floor(wy_min / ty) * ty
        while y <= wy_max + 1e-12:
            glVertex2f(wx_min, y); glVertex2f(wx_max, y)
            y += ty
        glEnd()

    def _draw_screen_axes_and_labels(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self.font)
        metrics = painter.fontMetrics()
        w, h = self.width(), self.height()

        # Colori Assi
        if self.view_mode == "N_Mx":
            c_x = self.axis_colors["Mx"]; lab_x = "Mx [kNm]"
            c_y = self.axis_colors["N"];  lab_y = "N [kN]"
        elif self.view_mode == "N_My":
            c_x = self.axis_colors["My"]; lab_x = "My [kNm]"
            c_y = self.axis_colors["N"];  lab_y = "N [kN]"
        else:
            c_x = self.axis_colors["Mx"]; lab_x = "Mx [kNm]"
            c_y = self.axis_colors["My"]; lab_y = "My [kNm]"

        # Helper
        def to_screen(wx, wy):
            nx = (wx - (-self.data_range_x * self.zoom_2d + self.pan_2d[0])) / (2 * self.data_range_x * self.zoom_2d)
            ny = (wy - (-self.data_range_y * self.zoom_2d + self.pan_2d[1])) / (2 * self.data_range_y * self.zoom_2d)
            return int(nx * w), int((1 - ny) * h)

        # 1. Disegna Linee Assi (X e Y passanti per origine)
        sx0, sy0 = to_screen(0, 0)
        
        # Asse X
        pen = painter.pen(); pen.setColor(c_x); pen.setWidth(1); painter.setPen(pen)
        painter.drawLine(0, sy0, w, sy0)
        
        # Asse Y
        pen.setColor(c_y); painter.setPen(pen)
        painter.drawLine(sx0, 0, sx0, h)

        # 2. Ticks e Valori
        wx_min = -self.data_range_x * self.zoom_2d + self.pan_2d[0]
        wx_max =  self.data_range_x * self.zoom_2d + self.pan_2d[0]
        wy_min = -self.data_range_y * self.zoom_2d + self.pan_2d[1]
        wy_max =  self.data_range_y * self.zoom_2d + self.pan_2d[1]
        
        tx = self._tick(wx_max - wx_min)
        ty = self._tick(wy_max - wy_min)

        # X Ticks
        painter.setPen(c_x)
        x = np.floor(wx_min / tx) * tx
        while x <= wx_max:
            if abs(x) > 1e-9:
                sx, sy = to_screen(x, 0)
                painter.drawLine(sx, sy-4, sx, sy+4)
                label = f"{x:.4g}"
                tw = metrics.horizontalAdvance(label)
                # Se asse X esce schermo vert, clamp label
                txt_y = sy + metrics.height() + 2
                if txt_y > h - 5: txt_y = h - 5
                if txt_y < 15: txt_y = 15
                painter.drawText(sx - tw//2, txt_y, label)
            x += tx

        # Y Ticks
        painter.setPen(c_y)
        y = np.floor(wy_min / ty) * ty
        while y <= wy_max:
            if abs(y) > 1e-9:
                sx, sy = to_screen(0, y)
                painter.drawLine(sx-4, sy, sx+4, sy)
                label = f"{y:.4g}"
                tw = metrics.horizontalAdvance(label)
                # Se asse Y esce schermo orizz, clamp label
                txt_x = sx - tw - 6
                if txt_x < 5: txt_x = 5
                if txt_x > w - tw - 5: txt_x = w - tw - 5
                painter.drawText(txt_x, sy + metrics.height()//2 - 2, label)
            y += ty

        # 3. Titoli Assi (Angoli)
        painter.setPen(c_x)
        tw = metrics.horizontalAdvance(lab_x)
        painter.drawText(w - tw - 10, h - 10, lab_x)

        painter.setPen(c_y)
        painter.save()
        painter.translate(20, 110)
        painter.rotate(-90)
        painter.drawText(0, 0, lab_y)
        painter.restore()

        painter.end()

    def _draw_tracker_with_coords(self):
        """Disegna cursore, linee di tracciamento e coordinate (Stile Reference)"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Usa il font impostato nel widget
        if hasattr(self, 'font_obj'):
            painter.setFont(self.font_obj)
        elif hasattr(self, 'font'):
            painter.setFont(self.font)

        w, h = self.width(), self.height()
        x, y = self.cursor_pos.x(), self.cursor_pos.y()

        # --- 1. DISEGNO LINEE MIRINO (FULL SCREEN) ---
        # Impostiamo un colore grigio discreto e semitrasparente
        tracking_pen = QPen(QColor(150, 150, 150, 100), 1, Qt.SolidLine)
        painter.setPen(tracking_pen)
        
        # Linea Verticale (da cima a fondo)
        painter.drawLine(x, 0, x, h)
        # Linea Orizzontale (da sinistra a destra)
        painter.drawLine(0, y, w, y)

        # --- 2. DISEGNO CROCE CENTRALE (PICCOLA) ---
        # Bianca e solida per indicare il punto esatto
        cross_pen = QPen(QColor(255, 255, 255, 255), 1)
        painter.setPen(cross_pen)
        size = 10
        painter.drawLine(x - size, y, x + size, y)
        painter.drawLine(x, y - size, x, y + size)

        # --- 3. CALCOLO E DISEGNO COORDINATE MONDO ---
        # Trasformiamo la posizione pixel in valori fisici
        wx, wy = self._screen_to_world(x, y)
        
        painter.setPen(QColor(255, 255, 255, 180)) # Testo bianco morbido
        lbl_x = f"{wx:.4g}"
        lbl_y = f"{wy:.4g}"

        # Posizionamento etichette (Stile professionale ai bordi)
        # X vicino al bordo inferiore, Y vicino al bordo sinistro
        painter.drawText(x + 10, h - 10, lbl_x)
        painter.drawText(10, y - 10, lbl_y)

        painter.end()

    # --- Utils Math ---
    def _tick(self, rng):
        rough = rng / 10
        if rough <= 0: return 1.0
        mag = 10 ** np.floor(np.log10(rough))
        res = rough / mag
        if res >= 5: return 5 * mag
        elif res >= 2: return 2 * mag
        return mag

    def _screen_to_world(self, sx, sy):
        w, h = self.width(), self.height()
        nx = sx / w
        ny = 1.0 - (sy / h)
        wx = (-self.data_range_x * self.zoom_2d + self.pan_2d[0]) + nx * (2 * self.data_range_x * self.zoom_2d)
        wy = (-self.data_range_y * self.zoom_2d + self.pan_2d[1]) + ny * (2 * self.data_range_y * self.zoom_2d)
        return wx, wy

    # --- Mouse ---
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.last_mouse_pos = e.pos()
        self.cursor_pos = e.pos()
        self.update()

    def mouseMoveEvent(self, e):
        self.cursor_pos = e.pos()
        if self.last_mouse_pos:
            dx = e.x() - self.last_mouse_pos.x()
            dy = e.y() - self.last_mouse_pos.y()
            
            # Pan in world units
            self.pan_2d[0] -= dx * (2 * self.data_range_x * self.zoom_2d) / self.width()
            self.pan_2d[1] += dy * (2 * self.data_range_y * self.zoom_2d) / self.height()
            
            self.last_mouse_pos = e.pos()
        self.update()

    def mouseReleaseEvent(self, e):
        self.last_mouse_pos = None
        self.update()

    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        if delta == 0: return
        
        # Zoom verso cursore
        sx, sy = e.pos().x(), e.pos().y()
        wx0, wy0 = self._screen_to_world(sx, sy)
        
        factor = 1.0 - np.sign(delta) * 0.1
        self.zoom_2d *= factor
        self.zoom_2d = max(0.001, min(100.0, self.zoom_2d))
        
        wx1, wy1 = self._screen_to_world(sx, sy)
        self.pan_2d[0] += (wx0 - wx1)
        self.pan_2d[1] += (wy0 - wy1)
        
        self.update()