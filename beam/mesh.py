# beam/mesh.py

import math
import traceback
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from PyQt5.QtWidgets import QOpenGLWidget, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer
# Importazioni necessarie per la legenda 2D
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QBrush, QLinearGradient

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    _HAS_GL = True
except Exception:
    _HAS_GL = False

# ------------------------------------------------------------------
# Dati semplici per voxel / barre (Architettonici)
# ------------------------------------------------------------------
@dataclass
class Voxel:
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    material: Optional[str]
    color: Tuple[float, float, float, float]        # RGBA
    vertices: List[Tuple[float, float, float]] = None
    faces: List[List[int]] = None
    grid_index: Tuple[int, int, int] = None

@dataclass
class ReinforcementBar:
    points: List[Tuple[float, float, float]]
    diameter: float
    material: Optional[str]
    color: Tuple[float, float, float, float]        # RGBA
    element_type: str = "long"

# ------------------------------------------------------------------
# Widget OpenGL
# ------------------------------------------------------------------
class BeamMeshWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # --- DATI ARCHITETTONICI ---
        self.concrete_voxels: List[Voxel] = []
        self.bars: List[ReinforcementBar] = []
        self.stirrups: List[ReinforcementBar] = []

        # Visibilità Architettonica
        self.show_concrete = True
        self.show_bars = True
        self.show_stirrups = True
        self.show_faces = True
        self.show_wireframe = True

        # --- DATI FEM ---
        self.fem_mode = False           # True se stiamo visualizzando i risultati FEM
        self.fem_result_type = 'disp'   # 'disp' (Deformazioni) o 'stress' (Sollecitazioni)
        
        # Storie temporali (Liste di numpy arrays)
        self.fem_disp_history = []      # Spostamenti [Step][Nodi][3]
        self.fem_stress_history = []    # Sollecitazioni [Step][Nodi] (scalari, es. Von Mises con segno)
        
        self.fem_coords = None          # Coordinate nodali indeformate [Nodi][3]
        self.fem_solid_elems = []       # Elementi HEX8
        self.fem_bar_elems = []         # Elementi TRUSS
        
        # Massimi globali (di tutta la storia)
        self.max_disp_global = 1.0            
        self.max_stress_global = 1.0          
        self.deformation_scale = 1.0    # Scala visiva della deformata

        # SCALA DINAMICA (Minimo e Massimo visibile nel frame corrente)
        self.current_visible_max = 1.0
        self.current_visible_min = 0.0  # <--- NUOVO: Gestione minimi (Trazione)

        # Cache indici nodi per categoria (per calcolo veloce scala)
        self._nodes_concrete_indices = []
        self._nodes_bars_indices = []
        self._nodes_stirrups_indices = []

        # --- ANIMAZIONE FEM ---
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._update_animation)
        self.anim_frame = 0.0
        self.anim_speed = 0.2
        self.is_animating = False

        # --- CAMERA ---
        self.rot_x = 30.0   
        self.rot_y = -45.0  
        self.distance = 5
        self.ortho_zoom = 1.5
        
        self.pan_x = 0
        self.pan_y = 0
        self.last_pos = None

        self.view_mode = '3d'
        self.setMinimumSize(200, 150)
        self.setFocusPolicy(Qt.StrongFocus) 

        self._ortho_default_zoom = 0.2
        self._background_color = (40.0 / 255.0, 40.0 / 255.0, 40.0 / 255.0, 1.0)

        self.update()

    def initializeGL(self):
        if not _HAS_GL:
            return
        glClearColor(*self._background_color)
        glEnable(GL_DEPTH_TEST)
        
        # Luci
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLightfv(GL_LIGHT0, GL_POSITION, [0.6, 0.6, 0.6, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.6, 0.6, 0.6, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.6, 0.6, 0.6, 1.0])

        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

    def resizeGL(self, w, h):
        if not _HAS_GL:
            return
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / max(h, 1)
        if self.view_mode == '3d':
            gluPerspective(45.0, aspect, 0.01, 100.0)
        else:
            left = -self.ortho_zoom * aspect
            right = self.ortho_zoom * aspect
            bottom = -self.ortho_zoom
            top = self.ortho_zoom
            glOrtho(left, right, bottom, top, -100.0, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        if not _HAS_GL:
            return
        
        # Riabilita Depth Test per evitare problemi con QPainter
        glEnable(GL_DEPTH_TEST) 
        glDepthMask(GL_TRUE)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Camera
        if self.view_mode == '3d':
            glTranslatef(self.pan_x, self.pan_y, -self.distance)
            glRotatef(self.rot_x, 1, 0, 0)
            glRotatef(self.rot_y, 0, 1, 0)
        else:
            glTranslatef(self.pan_x, self.pan_y, 0.0)
            if self.view_mode == 'yz': glRotatef(90.0, 0, 1, 0)
            elif self.view_mode == 'xz': glRotatef(-90.0, 1, 0, 0)
            elif self.view_mode == 'xy': pass

        self._draw_axes()

        if self.fem_mode and self.fem_coords is not None:
            self._draw_fem_results()
        else:
            # Vista Architettonica Standard
            if self.show_concrete and self.show_faces:
                self._draw_all_voxel_faces_opaque()
            
            if self.show_bars:
                for bar in self.bars: self._draw_reinforcement_bar_std(bar)
            if self.show_stirrups:
                for st in self.stirrups: self._draw_reinforcement_bar_std(st)
            
            if self.show_concrete and self.show_wireframe:
                self._draw_all_voxel_wireframes()

    # --- DISEGNO 2D OVERLAY (LEGENDA) ---
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.fem_mode:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            self._draw_legend_overlay(painter)
            painter.end()

    def _draw_legend_overlay(self, painter: QPainter):
        """Disegna la scala dei colori con gestione range Min-Max (Trazione-Compressione)"""
        w = self.width()
        h = self.height()
        
        # Dimensioni e posizione legenda
        bar_width = 50
        bar_height = min(700, h - 60)
        margin_right = 110 
        x_pos = w - margin_right - bar_width
        y_pos = (h - bar_height) // 2
        
        # Sfondo
        bg_rect_x = x_pos - 10
        bg_rect_y = y_pos - 25
        bg_rect_w = bar_width + margin_right - 30
        bg_rect_h = bar_height + 50
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(25,25,25, 180))
        painter.drawRoundedRect(bg_rect_x, bg_rect_y, bg_rect_w, bg_rect_h, 5, 5)

        # Valori di riferimento MIN e MAX
        max_val = self.current_visible_max
        min_val = self.current_visible_min

        # Se è 'disp' (magnitudo), il minimo è sempre 0
        if self.fem_result_type == 'disp':
            min_val = 0.0
            title = "Spost. [m]"
        else:
            title = "Stress [Pa]"

        val_range = max_val - min_val
        if abs(val_range) < 1e-12: val_range = 1.0

        # Titolo
        painter.setPen(Qt.white)
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(bg_rect_x, bg_rect_y, bg_rect_w, -45, Qt.AlignCenter, title)

        # Gradiente
        gradient = QLinearGradient(x_pos, y_pos + bar_height, x_pos, y_pos) 
        
        if self.fem_result_type == 'disp':
            # Blu (0) -> Rosso (Max)
            gradient.setColorAt(0.0, QColor(0, 0, 255))
            gradient.setColorAt(0.5, QColor(255, 0, 255))
            gradient.setColorAt(1.0, QColor(255, 0, 0))
        else:
            # Jet Map Completa: Blu (Min/Trazione) -> Rosso (Max/Compressione)
            gradient.setColorAt(0.00, QColor(0, 0, 255))
            gradient.setColorAt(0.25, QColor(0, 255, 255))
            gradient.setColorAt(0.50, QColor(0, 255, 0))
            gradient.setColorAt(0.75, QColor(255, 255, 0))
            gradient.setColorAt(1.00, QColor(255, 0, 0))

        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(Qt.white, 1))
        painter.drawRect(x_pos, y_pos, bar_width, bar_height)

        # Etichette numeriche
        font.setBold(False)
        font.setPointSize(8)
        painter.setFont(font)
        
        num_ticks = 5
        for i in range(num_ticks):
            ratio = i / (num_ticks - 1) 
            
            # Interpolazione lineare tra Min e Max
            val = min_val + ratio * val_range
            
            y_tick = y_pos + bar_height - (ratio * bar_height)
            
            painter.drawLine(x_pos + bar_width, int(y_tick), x_pos + bar_width + 5, int(y_tick))
            
            if abs(val) > 1000 or (abs(val) < 0.01 and abs(val) > 1e-12):
                text_str = f"{val:.2e}"
            else:
                text_str = f"{val:.2f}"
                
            painter.drawText(x_pos + bar_width + 8, int(y_tick) - 10, 60, 20, Qt.AlignLeft | Qt.AlignVCenter, text_str)

    # ------------------------------------------------------------------
    # Helpers Disegno
    # ------------------------------------------------------------------
    def _draw_axes(self):
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor3f(1, 0, 0); glVertex3f(0, 0, 0); glVertex3f(1, 0, 0)
        glColor3f(0, 1, 0); glVertex3f(0, 0, 0); glVertex3f(0, 1, 0)
        glColor3f(0, 0, 1); glVertex3f(0, 0, 0); glVertex3f(0, 0, 1)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_all_voxel_faces_opaque(self):
        if not self.concrete_voxels: return
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)
        glEnable(GL_LIGHTING)
        for voxel in self.concrete_voxels:
            if not voxel.faces: continue
            glColor4f(*voxel.color)
            glBegin(GL_QUADS)
            for face in voxel.faces:
                for vid in face:
                    glVertex3f(*voxel.vertices[vid])
            glEnd()
        glDisable(GL_POLYGON_OFFSET_FILL)

    def _draw_all_voxel_wireframes(self):
        if not self.concrete_voxels: return
        glDisable(GL_LIGHTING)
        glColor3f(0.10, 0.10, 0.10)
        glLineWidth(1.0)
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        glBegin(GL_LINES)
        for voxel in self.concrete_voxels:
            for a,b in edges:
                glVertex3f(*voxel.vertices[a]); glVertex3f(*voxel.vertices[b])
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_reinforcement_bar_std(self, bar: ReinforcementBar):
        if len(bar.points) < 2: return
        glDisable(GL_LIGHTING)
        glColor4f(*bar.color)
        glLineWidth(2.5 if bar.element_type == "stirrup" else 2.0)
        mode = GL_LINE_STRIP
        if bar.element_type == "stirrup" and len(bar.points) > 2 and bar.points[0] == bar.points[-1]:
            mode = GL_LINE_LOOP
        glBegin(mode)
        for p in bar.points:
            glVertex3f(*p)
        glEnd()
        glEnable(GL_LIGHTING)

     # ------------------------------------------------------------------
    # GESTIONE FEM
    # ------------------------------------------------------------------
    def set_fem_results(self, 
                        disp_history: List[np.ndarray], 
                        coords: List[Tuple[float, float, float]], 
                        solid_elems: List[Dict], 
                        bar_elems: List[Dict], 
                        max_disp: float,
                        stress_history: Optional[List[np.ndarray]] = None,
                        max_stress: float = 1.0):
        
        self.fem_disp_history = disp_history
        self.fem_stress_history = stress_history if stress_history is not None else []
        self.fem_coords = np.array(coords)
        self.fem_solid_elems = solid_elems
        self.fem_bar_elems = bar_elems
        
        # Salviamo i globali come fallback
        self.max_disp_global = max_disp if max_disp > 1e-9 else 1.0
        self.max_stress_global = max_stress if max_stress > 1e-9 else 1.0
        
        # --- PRE-CALCOLO INDICI PER CATEGORIA (Per scala dinamica) ---
        c_nodes = set()
        for el in solid_elems:
            c_nodes.update(el['nodes'])
        self._nodes_concrete_indices = list(c_nodes)

        b_nodes = set()
        s_nodes = set()
        for el in bar_elems:
            # Assumiamo che il generatore usi 'TRUSS_LONG' e 'TRUSS_STIR'
            if el.get('type') == 'TRUSS_LONG':
                b_nodes.update(el['nodes'])
            else:
                s_nodes.update(el['nodes'])
        self._nodes_bars_indices = list(b_nodes)
        self._nodes_stirrups_indices = list(s_nodes)
        # -------------------------------------------------------------

        self.fem_mode = True
        self.start_animation()

    def set_result_type(self, r_type: str):
        if r_type in ['disp', 'stress']:
            self.fem_result_type = r_type
            self.update()

    # --- ANIMAZIONE ---
    def start_animation(self):
        self.is_animating = True
        # Resetto il frame solo se siamo alla fine
        if self.anim_frame >= len(self.fem_disp_history) - 1:
            self.anim_frame = 0.0
        self.animation_timer.start(100)


    def stop_animation(self):
        self.is_animating = False
        self.animation_timer.stop()

    def _update_animation(self):
        if not self.fem_disp_history:
            return
        self.anim_frame += self.anim_speed
        max_frame = len(self.fem_disp_history) - 1
        if self.anim_frame >= max_frame:
            self.anim_frame = float(max_frame)
            self.stop_animation()
        self.update()

    def keyPressEvent(self, event):
        if self.fem_mode and event.key() == Qt.Key_Space:
            if self.is_animating:
                self.stop_animation()     # Se sta animando → ferma
            else:
                self.start_animation()    # Se è ferma → avvia
            return                        # Evita di propagare l'evento
        super().keyPressEvent(event)


    

    # --- CALCOLO DATI FRAME CORRENTE ---
    def _get_frame_data(self):
        """
        Ritorna (coords_deformate, values_per_node, global_max)
        Il global_max qui è solo il riferimento, poi viene raffinato in draw
        """
        if not self.fem_disp_history:
            return self.fem_coords, np.zeros(len(self.fem_coords)), 1.0

        idx_floor = int(math.floor(self.anim_frame))
        idx_ceil = min(idx_floor + 1, len(self.fem_disp_history) - 1)
        alpha = self.anim_frame - idx_floor

        # Interpolazione Spostamenti
        u_curr = self.fem_disp_history[idx_floor] * (1.0 - alpha) + self.fem_disp_history[idx_ceil] * alpha
        deformed_coords = self.fem_coords + u_curr.reshape(-1, 3) * self.deformation_scale
        
        values = np.zeros(len(deformed_coords))
        global_ref = 1.0

        if self.fem_result_type == 'disp':
            # Magnitudo spostamento (Sempre positivo)
            values = np.linalg.norm(u_curr.reshape(-1, 3), axis=1)
            global_ref = self.max_disp_global
        else:
            # Stress (Può essere negativo)
            global_ref = self.max_stress_global
            if self.fem_stress_history and len(self.fem_stress_history) > 0:
                s_floor = self.fem_stress_history[idx_floor]
                s_ceil = self.fem_stress_history[idx_ceil]
                s_curr = s_floor * (1.0 - alpha) + s_ceil * alpha
                values = s_curr
            else:
                values = np.zeros(len(deformed_coords))

        return deformed_coords, values, global_ref

    # --- MAPPE COLORI ---
    def _get_color_disp(self, val, max_val):
        """ Deformazioni: Blu (0) -> Rosso (Max) """
        if max_val < 1e-12: max_val = 1.0
        t = val / max_val
        t = max(0.0, min(1.0, t))
        return (t, 0.0, 1.0 - t)

    def _get_color_stress(self, val, min_val, max_val):
        """ 
        Sollecitazioni: Jet Map che copre il range [min_val, max_val]
        Blu = min_val (es. -10 MPa)
        Rosso = max_val (es. +20 MPa)
        """
        denom = max_val - min_val
        if abs(denom) < 1e-12: denom = 1.0
        
        # Normalizza tra 0 e 1
        t = (val - min_val) / denom
        t = max(0.0, min(1.0, t))
        
        r, g, b = 0.0, 0.0, 0.0
        if t < 0.25:
            slope = t * 4.0
            r, g, b = 0.0, slope, 1.0
        elif t < 0.5:
            slope = (t - 0.25) * 4.0
            r, g, b = 0.0, 1.0, 1.0 - slope
        elif t < 0.75:
            slope = (t - 0.5) * 4.0
            r, g, b = slope, 1.0, 0.0
        else:
            slope = (t - 0.75) * 4.0
            r, g, b = 1.0, 1.0 - slope, 0.0
        return (r, g, b)

    def _draw_fem_results(self):
        # 1. Recupera dati frame corrente
        deformed_coords, values, global_norm = self._get_frame_data()
        
        # 2. CALCOLO SCALA ADATTIVA (DYNAMIC RANGE)
        # Identifica quali nodi sono visibili al momento
        active_indices = []
        if self.show_concrete:
            active_indices.extend(self._nodes_concrete_indices)
        if self.show_bars:
            active_indices.extend(self._nodes_bars_indices)
        if self.show_stirrups:
            active_indices.extend(self._nodes_stirrups_indices)
            
        if active_indices and len(values) > 0:
            # Filtra indici validi e unici
            idx_arr = np.array(active_indices, dtype=int)
            idx_arr = idx_arr[idx_arr < len(values)]
            
            if len(idx_arr) > 0:
                # Calcola MIN e MAX tra i nodi visibili
                vis_max = np.max(values[idx_arr])
                vis_min = np.min(values[idx_arr])
                
                self.current_visible_max = vis_max
                self.current_visible_min = vis_min
            else:
                self.current_visible_max = global_norm
                self.current_visible_min = 0.0
        else:
            self.current_visible_max = global_norm
            self.current_visible_min = 0.0

        max_v = self.current_visible_max
        min_v = self.current_visible_min
        # --------------------------------------------

        def get_color(v):
            if self.fem_result_type == 'disp':
                # Per gli spostamenti, il minimo è sempre 0
                return self._get_color_disp(v, max_v)
            else:
                # Per gli stress, passiamo sia min che max
                return self._get_color_stress(v, min_v, max_v)

        faces_idx = [
            [0,1,2,3], [4,5,6,7], [0,1,5,4], 
            [1,2,6,5], [2,3,7,6], [3,0,4,7]
        ]

        # 1) Solidi
        if self.show_concrete:
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(1.0, 1.0)
            glEnable(GL_LIGHTING)
            
            glBegin(GL_QUADS)
            for el in self.fem_solid_elems:
                ns = el['nodes']
                for face in faces_idx:
                    for idx in face:
                        node_idx = ns[idx]
                        c = get_color(values[node_idx])
                        glColor3f(*c)
                        p = deformed_coords[node_idx]
                        glVertex3f(p[0], p[1], p[2])
            glEnd()
            glDisable(GL_POLYGON_OFFSET_FILL)

            if self.show_wireframe:
                glDisable(GL_LIGHTING)
                glColor3f(0.2, 0.2, 0.2)
                glLineWidth(1.0)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glBegin(GL_QUADS)
                for el in self.fem_solid_elems:
                    ns = el['nodes']
                    for face in faces_idx:
                        for idx in face:
                            p = deformed_coords[ns[idx]]
                            glVertex3f(p[0], p[1], p[2])
                glEnd()
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # 2) Barre e Staffe
        glEnable(GL_LIGHTING)
        glLineWidth(2.5)
        glBegin(GL_LINES)
        for b in self.fem_bar_elems:
            is_long = (b.get('type') == 'TRUSS_LONG')
            if is_long and not self.show_bars: continue
            if not is_long and not self.show_stirrups: continue

            n1, n2 = b['nodes']
            p1 = deformed_coords[n1]
            p2 = deformed_coords[n2]
            
            c1 = get_color(values[n1])
            glColor3f(*c1)
            glVertex3f(*p1)
            
            c2 = get_color(values[n2])
            glColor3f(*c2)
            glVertex3f(*p2)
        glEnd()

    # --- MOUSE INTERACTION ---
    def mousePressEvent(self, event):
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            self.last_pos = event.pos()
            return
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()

        if self.view_mode == '3d':
            if event.buttons() & Qt.LeftButton:
                self.rot_x += dy * 0.5
                self.rot_y += dx * 0.5
            elif event.buttons() & Qt.RightButton:
                pan_scale = (0.005 * max(1.0, self.distance))
                self.pan_x += dx * pan_scale
                self.pan_y -= dy * pan_scale
        else:
            if event.buttons() & Qt.LeftButton:
                denom = max(1, min(self.width(), self.height()))
                pan_scale = (2.0 * self.ortho_zoom) / denom
                self.pan_x += dx * pan_scale
                self.pan_y -= dy * pan_scale
            elif event.buttons() & Qt.RightButton:
                factor = 1.0 - dy * 0.01
                factor = max(0.1, min(10.0, factor))
                self.ortho_zoom *= factor
                self.ortho_zoom = max(0.001, min(self.ortho_zoom, 100.0))
                self.resizeGL(self.width(), self.height())

        self.last_pos = event.pos()
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120.0
        if self.view_mode == '3d':
            self.distance -= delta * 0.5
            self.distance = max(0.1, min(self.distance, 100.0))
        else:
            factor = 1.0 - delta * 0.1
            factor = max(0.01, min(10.0, factor))
            self.ortho_zoom *= factor
            self.ortho_zoom = max(0.01, min(self.ortho_zoom, 100.0))
            self.resizeGL(self.width(), self.height())
        self.update()

    # --- API SETTERS ---
    def set_mesh_data(self, concrete_voxels: List[Voxel], bars: List[ReinforcementBar], stirrups: List[ReinforcementBar]):
        self.concrete_voxels = concrete_voxels or []
        self.bars = bars or []
        self.stirrups = stirrups or []
        self.fem_mode = False
        self.stop_animation()
        self.update()

    def set_view_mode(self, mode: str):
        if mode not in ('3d', 'yz', 'xz', 'xy'): return
        if mode == '3d':
            self.view_mode = '3d'
            self.distance = max(1.0, self.distance)
        else:
            self.view_mode = mode
            self.rot_x = 0.0
            self.rot_y = 0.0
            self.ortho_zoom = max(0.001, self._ortho_default_zoom)
        self.resizeGL(self.width(), self.height())
        self.update()

# ------------------------------------------------------------------
# Core generator
# ------------------------------------------------------------------
class BeamMeshCore:
    def __init__(self):
        pass

    def _mm_to_m(self, mm_val):
        return float(mm_val) / 1000.0

    def _point_in_polygon(self, x, y, poly):
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _point_in_circle(self, x, y, center, radius):
        cx, cy = center
        return (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

    def _get_section_bounding_box(self, section):
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')

        for rect in section.get('rects', []):
            v1 = rect.get('v1', (0, 0)); v2 = rect.get('v2', (0, 0))
            for x, y in (v1, v2):
                min_x = min(min_x, x); max_x = max(max_x, x)
                min_y = min(min_y, y); max_y = max(max_y, y)

        for circle in section.get('circles', []):
            center = circle.get('center', (0, 0))
            radius_point = circle.get('radius_point', (0, 0))
            radius = math.hypot(center[0] - radius_point[0], center[1] - radius_point[1])
            cx, cy = center
            min_x = min(min_x, cx - radius); max_x = max(max_x, cx + radius)
            min_y = min(min_y, cy - radius); max_y = max(max_y, cy + radius)

        for poly in section.get('polys', []):
            for x, y in poly.get('vertices', []):
                min_x = min(min_x, x); max_x = max(max_x, x)
                min_y = min(min_y, y); max_y = max(max_y, y)

        if min_x == float('inf'):
            return 0.0, 0.0, 0.0, 0.0
        return min_x, max_x, min_y, max_y

    def _is_point_in_section(self, x, y, section):
        for rect in section.get('rects', []):
            v1 = rect.get('v1', (0, 0)); v2 = rect.get('v2', (0, 0))
            x1, y1 = min(v1[0], v2[0]), min(v1[1], v2[1])
            x2, y2 = max(v1[0], v2[0]), max(v1[1], v2[1])
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True, rect.get('material')

        for circle in section.get('circles', []):
            center = circle.get('center', (0, 0))
            rpoint = circle.get('radius_point', (0, 0))
            radius = math.hypot(center[0] - rpoint[0], center[1] - rpoint[1])
            if self._point_in_circle(x, y, center, radius):
                return True, circle.get('material')

        for poly_obj in section.get('polys', []):
            vertices = poly_obj.get('vertices', [])
            if len(vertices) >= 3 and self._point_in_polygon(x, y, vertices):
                return True, poly_obj.get('material')
        return False, None

    def _create_hexahedron_vertices(self, center, size):
        cx, cy, cz = center
        sx, sy, sz = size
        vertices = [
            (cx - sx / 2, cy - sy / 2, cz - sz / 2), (cx + sx / 2, cy - sy / 2, cz - sz / 2),
            (cx + sx / 2, cy + sy / 2, cz - sz / 2), (cx - sx / 2, cy + sy / 2, cz - sz / 2),
            (cx - sx / 2, cy - sy / 2, cz + sz / 2), (cx + sx / 2, cy - sy / 2, cz + sz / 2),
            (cx + sx / 2, cy + sy / 2, cz + sz / 2), (cx - sx / 2, cy + sy / 2, cz + sz / 2)
        ]
        faces = [
            [0,1,2,3], [4,5,6,7], [0,1,5,4], 
            [1,2,6,5], [2,3,7,6], [3,0,4,7]
        ]
        return vertices, faces

    def generate_concrete_mesh(self, section: Dict[str, Any], length_m: float, subdivisions_x: int, subdivisions_y: int, subdivisions_z: int) -> List[Voxel]:
        voxels: List[Voxel] = []
        min_x, max_x, min_y, max_y = self._get_section_bounding_box(section)
        if max_x - min_x < 1e-9 or max_y - min_y < 1e-9: return voxels

        dx_mm = (max_x - min_x) / max(1, subdivisions_x)
        dy_mm = (max_y - min_y) / max(1, subdivisions_y)
        dz_m = length_m / max(1, subdivisions_z)

        body_gray = (0.65, 0.65, 0.65, 1.0)
        inside_2d = [[False for _ in range(subdivisions_y)] for __ in range(subdivisions_x)]
        materials_2d = [[None for _ in range(subdivisions_y)] for __ in range(subdivisions_x)]
        
        for i in range(subdivisions_x):
            for j in range(subdivisions_y):
                x_mm = min_x + (i + 0.5) * dx_mm
                y_mm = min_y + (j + 0.5) * dy_mm
                is_inside, material = self._is_point_in_section(x_mm, y_mm, section)
                inside_2d[i][j] = is_inside
                materials_2d[i][j] = material

        for i in range(subdivisions_x):
            for j in range(subdivisions_y):
                if not inside_2d[i][j]: continue
                for k in range(subdivisions_z):
                    z_m = (k + 0.5) * dz_m
                    center = (self._mm_to_m(min_x + (i + 0.5) * dx_mm), self._mm_to_m(min_y + (j + 0.5) * dy_mm), z_m)
                    size = (self._mm_to_m(dx_mm), self._mm_to_m(dy_mm), dz_m)
                    vertices, full_faces = self._create_hexahedron_vertices(center, size)

                    neighbor_offsets = [ (0,0,-1), (0,0,1), (0,-1,0), (1,0,0), (0,1,0), (-1,0,0) ]
                    faces_external = []
                    for fi, off in enumerate(neighbor_offsets):
                        ni, nj, nk = i + off[0], j + off[1], k + off[2]
                        if 0 <= ni < subdivisions_x and 0 <= nj < subdivisions_y and 0 <= nk < subdivisions_z:
                            neighbor_inside = inside_2d[ni][nj]
                        else:
                            neighbor_inside = False
                        if not neighbor_inside:
                            faces_external.append(full_faces[fi])
                    voxels.append(Voxel(center=center, size=size, material=materials_2d[i][j], color=body_gray, vertices=vertices, faces=faces_external, grid_index=(i,j,k)))
        return voxels

    def generate_bars_mesh(self, section: Dict[str, Any], length_m: float, subdivisions_z: int = 1, spacing_m: float = 0.0) -> List[ReinforcementBar]:
        bars: List[ReinforcementBar] = []
        bar_red = (1.0, 0.0, 0.0, 1.0)
        
        if spacing_m is not None and spacing_m > 0.0:
            num_segments = max(1, int(math.ceil(length_m / spacing_m)))
            use_spacing = True
        else:
            num_segments = max(1, int(max(1, subdivisions_z)))
            use_spacing = False

        for bar in section.get('bars', []):
            center_2d = bar.get('center', (0, 0))
            diameter = bar.get('diam', 10)
            material = bar.get('material')
            x = self._mm_to_m(center_2d[0]); y = self._mm_to_m(center_2d[1])
            diam_m = self._mm_to_m(diameter)

            if use_spacing:
                for i in range(num_segments):
                    z1 = i * spacing_m
                    z2 = min((i + 1) * spacing_m, length_m)
                    if z2 <= z1: continue
                    bars.append(ReinforcementBar(points=[(x, y, z1), (x, y, z2)], diameter=diam_m, material=material, color=bar_red, element_type="long"))
            else:
                seg_len = length_m / float(num_segments)
                for i in range(num_segments):
                    z1 = i * seg_len; z2 = (i + 1) * seg_len
                    bars.append(ReinforcementBar(points=[(x, y, z1), (x, y, z2)], diameter=diam_m, material=material, color=bar_red, element_type="long"))
        return bars

    def generate_stirrups_mesh(self, section: Dict[str, Any], length_m: float, spacing_m: float = 0.0) -> List[ReinforcementBar]:
        stirrups: List[ReinforcementBar] = []
        for stirrup in section.get('staffe', []):
            points_2d = stirrup.get('points', [])
            diameter = stirrup.get('diam', 8)
            material = stirrup.get('material')
            if len(points_2d) < 2: continue
            points_m = [(self._mm_to_m(x), self._mm_to_m(y)) for x, y in points_2d]
            diam_m = self._mm_to_m(diameter)
            
            if spacing_m is None or spacing_m <= 0.0:
                num = 1; step = length_m
            else:
                num = max(1, int(math.ceil(length_m / spacing_m)))
                step = spacing_m

            for i in range(num):
                z = (i + 0.5) * step
                if z > length_m: continue
                pts3d = [(x, y, z) for x, y in points_m]
                if len(pts3d) > 2 and pts3d[0] != pts3d[-1]: pts3d.append(pts3d[0])
                stirrups.append(ReinforcementBar(points=pts3d, diameter=diam_m, material=material, color=(1.0, 1.0, 0.0, 1.0), element_type="stirrup"))
        return stirrups

# ------------------------------------------------------------------
# Controller
# ------------------------------------------------------------------
class BeamMeshGenerator:
    def __init__(self, parent, ui, section_manager, gestione_materiali):
        self.parent = parent
        self.ui = ui
        self.section_manager = section_manager
        self.gestione_materiali = gestione_materiali

        try:
            from beam.valori import BeamValori
            self.beam_valori = BeamValori(self.section_manager, ui=self.ui, gestione_materiali=self.gestione_materiali)
        except Exception:
            self.beam_valori = None

        self.core = BeamMeshCore()
        self.selected_section_index = None

        try: self.ui.beam_mesh.clicked.connect(self.generate_mesh)
        except Exception: pass

        self.gl_widget = None
        self._ensure_gl_widget_in_ui()

        # Checkbox Visibilità
        try:
            if hasattr(self.ui, 'btn_beam_corpo'):
                self.ui.btn_beam_corpo.setCheckable(True); self.ui.btn_beam_corpo.setChecked(True)
                self.ui.btn_beam_corpo.toggled.connect(self._on_toggle_concrete)
            if hasattr(self.ui, 'btn_beam_barre'):
                self.ui.btn_beam_barre.setCheckable(True); self.ui.btn_beam_barre.setChecked(True)
                self.ui.btn_beam_barre.toggled.connect(self._on_toggle_bars)
            if hasattr(self.ui, 'btn_beam_staffe'):
                self.ui.btn_beam_staffe.setCheckable(True); self.ui.btn_beam_staffe.setChecked(True)
                self.ui.btn_beam_staffe.toggled.connect(self._on_toggle_stirrups)
        except Exception: pass

        # Pulsanti Vista
        try:
            if hasattr(self.ui, 'btn_beam_3d'): self.ui.btn_beam_3d.clicked.connect(lambda: self._set_view_mode_safe('3d'))
            if hasattr(self.ui, 'btn_beam_yz'): self.ui.btn_beam_yz.clicked.connect(lambda: self._set_view_mode_safe('yz'))
            if hasattr(self.ui, 'btn_beam_xz'): self.ui.btn_beam_xz.clicked.connect(lambda: self._set_view_mode_safe('xz'))
            if hasattr(self.ui, 'btn_beam_xy'): self.ui.btn_beam_xy.clicked.connect(lambda: self._set_view_mode_safe('xy'))
        except Exception: pass

        # --- PULSANTI MODALITÀ FEM (Esclusivi) ---
        try:
            if hasattr(self.ui, 'btn_beam_deformazioni'):
                self.ui.btn_beam_deformazioni.setCheckable(True)
                self.ui.btn_beam_deformazioni.setChecked(True) # Default
                self.ui.btn_beam_deformazioni.clicked.connect(self._on_click_deformazioni)
            
            if hasattr(self.ui, 'btn_beam_sollecitazioni'):
                self.ui.btn_beam_sollecitazioni.setCheckable(True)
                self.ui.btn_beam_sollecitazioni.setChecked(False)
                self.ui.btn_beam_sollecitazioni.clicked.connect(self._on_click_sollecitazioni)
        except Exception: pass

    def _on_click_deformazioni(self):
        # Logica Radio Button
        if hasattr(self.ui, 'btn_beam_deformazioni'): self.ui.btn_beam_deformazioni.setChecked(True)
        if hasattr(self.ui, 'btn_beam_sollecitazioni'): self.ui.btn_beam_sollecitazioni.setChecked(False)
        
        if self.gl_widget:
            self.gl_widget.set_result_type('disp')

    def _on_click_sollecitazioni(self):
        # Logica Radio Button
        if hasattr(self.ui, 'btn_beam_deformazioni'): self.ui.btn_beam_deformazioni.setChecked(False)
        if hasattr(self.ui, 'btn_beam_sollecitazioni'): self.ui.btn_beam_sollecitazioni.setChecked(True)
        
        if self.gl_widget:
            self.gl_widget.set_result_type('stress')

    def _resolve_material_from_obj(self, obj, mats) -> Optional[str]:
        mat_names = []
        try:
            for m in mats:
                if isinstance(m, (list, tuple)) and len(m) > 0: mat_names.append(str(m[0]))
                else: mat_names.append(str(m))
        except Exception: mat_names = []

        if isinstance(obj, (list, tuple)):
            last = obj[-1]
            if isinstance(last, str) and last.strip(): return last
            if isinstance(last, int) and 0 <= last < len(mat_names): return mat_names[last]
            for el in obj:
                if isinstance(el, str) and el in mat_names: return el
            for el in obj:
                if isinstance(el, int) and 0 <= el < len(mat_names): return mat_names[el]
        return None

    def _build_section_from_matrices(self, mats, objs) -> dict:
        section = {'rects': [], 'circles': [], 'polys': [], 'bars': [], 'staffe': []}
        for obj in objs:
            if not obj: continue
            tag = str(obj[0]).upper()

            # CERCHIO
            if 'CERCHIO' in tag or 'CIRCLE' in tag:
                if len(obj) >= 4: center = obj[1]; radius_point = obj[2]; material = obj[3]
                elif len(obj) >= 3: center = obj[1]; radius_point = obj[2]; material = self._resolve_material_from_obj(obj, mats)
                else: continue
                section['circles'].append({'center': center, 'radius_point': radius_point, 'material': material})
                continue

            # RETTANGOLO
            if 'RETT' in tag or 'RECT' in tag:
                if len(obj) >= 4: v1 = obj[1]; v2 = obj[2]; material = obj[3]
                elif len(obj) >= 3: v1 = obj[1]; v2 = obj[2]; material = self._resolve_material_from_obj(obj, mats)
                else: continue
                section['rects'].append({'v1': v1, 'v2': v2, 'material': material})
                continue

            # POLIGONO
            if 'POLIG' in tag or 'POLYGON' in tag:
                if len(obj) >= 3:
                    *pts, maybe_mat = obj[1:]
                    if isinstance(maybe_mat, str) and len(pts) >= 1:
                        vertices = [tuple(pt) for pt in pts]; material = maybe_mat
                    else:
                        vertices = [tuple(x) for x in obj[1:]]; material = self._resolve_material_from_obj(obj, mats)
                    section['polys'].append({'vertices': vertices, 'material': material})
                continue

            # BARRE (B / BAR)
            if tag.startswith('B') or 'BAR' in tag:
                if len(obj) >= 4: center = obj[1]; diam = obj[2]; material = obj[3]
                elif len(obj) >= 3:
                    center = obj[1]; diam = obj[2] if isinstance(obj[2], (int, float)) else 10
                    material = self._resolve_material_from_obj(obj, mats)
                elif len(obj) >= 2:
                    center = obj[1]; diam = 10; material = self._resolve_material_from_obj(obj, mats)
                else: continue
                section['bars'].append({'center': center, 'diam': diam, 'material': material})
                continue

            # STAFFE
            if tag.startswith('S') or 'STAFFA' in tag or 'STAFFE' in tag:
                if len(obj) >= 3:
                    material = None; diam = None
                    rest = list(obj[1:])
                    if isinstance(rest[-1], str): material = rest[-1]; rest = rest[:-1]
                    if len(rest) > 0 and isinstance(rest[-1], (int, float)): diam = rest[-1]; rest = rest[:-1]
                    if material is None: material = self._resolve_material_from_obj(obj, mats)
                    if diam is None: diam = 8
                    points = [tuple(p) for p in rest]
                    section['staffe'].append({'points': points, 'diam': diam, 'material': material})
                continue
            
            # Fallback
            if len(obj) >= 3 and isinstance(obj[1], (tuple, list)):
                center = obj[1]
                diam = obj[2] if isinstance(obj[2], (int, float)) else 10
                material = obj[3] if len(obj) > 3 and isinstance(obj[3], str) else self._resolve_material_from_obj(obj, mats)
                section['bars'].append({'center': center, 'diam': diam, 'material': material})

        return section

    def _ensure_gl_widget_in_ui(self):
        container_widget = getattr(self.ui, 'widget_beam', None)
        if container_widget is None: return None
        if self.gl_widget is not None: return self.gl_widget
        
        self.gl_widget = BeamMeshWidget(parent=container_widget)
        layout = container_widget.layout()
        if layout is None:
            layout = QVBoxLayout(container_widget)
            layout.setContentsMargins(1,1,1,1)
            container_widget.setLayout(layout)
        
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None: w.setParent(None)
        
        layout.addWidget(self.gl_widget)
        self.gl_widget.update()
        return self.gl_widget

    def _on_toggle_concrete(self, checked: bool):
        if self.gl_widget: self.gl_widget.show_concrete = checked; self.gl_widget.update()
    def _on_toggle_bars(self, checked: bool):
        if self.gl_widget: self.gl_widget.show_bars = checked; self.gl_widget.update()
    def _on_toggle_stirrups(self, checked: bool):
        if self.gl_widget: self.gl_widget.show_stirrups = checked; self.gl_widget.update()
    def _set_view_mode_safe(self, mode: str):
        if self.gl_widget: self.gl_widget.set_view_mode(mode)

    def generate_mesh(self):
        if self.beam_valori is None: return

        # 1) Indice sezione
        sel_idx = None
        combo = getattr(self.ui, 'combobox_beam_sezioni', None)
        if combo is not None:
            try:
                ci = combo.currentIndex()
                data = combo.itemData(ci, Qt.UserRole)
                sel_idx = int(data) if data is not None else int(ci)
            except Exception: sel_idx = None

        if sel_idx is None and self.selected_section_index is not None:
            sel_idx = int(self.selected_section_index)
        if sel_idx is None: sel_idx = 0
        self.selected_section_index = sel_idx

        # 2) Matrici
        try:
            mats, objs = self.beam_valori.generate_matrices(sel_idx)
        except Exception: return

        # 3) Parametri UI
        try: lunghezza = float(self.ui.beam_lunghezza.text())
        except: lunghezza = 1.0
        try: passo_staffe = float(self.ui.beam_passo.text())
        except: passo_staffe = 0.0
        try: nx = int(self.ui.beam_definizione_x.text())
        except: nx = 14
        try: ny = int(self.ui.beam_definizione_y.text())
        except: ny = 14
        try: nz = int(self.ui.beam_definizione_z.text())
        except: nz = 16

        # 4) Costruzione
        section = self._build_section_from_matrices(mats, objs)

        # 5) Generazione Mesh Architettonica
        concrete_voxels = self.core.generate_concrete_mesh(section, lunghezza, nx, ny, nz)
        bars = self.core.generate_bars_mesh(section, lunghezza, subdivisions_z=nz, spacing_m=0.0)
        stirrups = self.core.generate_stirrups_mesh(section, lunghezza, spacing_m=passo_staffe if passo_staffe > 0.0 else 0.0)

        # 6) Caricamento
        glw = self._ensure_gl_widget_in_ui()
        if glw:
            glw.set_mesh_data(concrete_voxels, bars, stirrups)
            glw.set_view_mode('3d')
            glw.update()