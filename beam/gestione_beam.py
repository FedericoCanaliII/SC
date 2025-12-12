from typing import Any, Optional
import traceback
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QButtonGroup, QMessageBox, QComboBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import QtCore

# Importa le classi mesh
from beam.valori import BeamValori
from beam.mesh import BeamMeshGenerator
# Aggiungi all'import
from beam.calcolo import BeamCalcolo
from beam.calcolo_dg import BeamCalcolo_dg


class GestioneBeam:
    def __init__(self, parent, ui, gestione_sezioni, gestione_materiali):
        self.ui = ui
        self.parent = parent
        self.gestione_sezioni = gestione_sezioni
        self.gestione_materiali = gestione_materiali

        self.beam_valori = BeamValori(
            getattr(self.gestione_sezioni, 'section_manager', self.gestione_sezioni),
            ui=self.ui,
            gestione_materiali=self.gestione_materiali
        )

        self.beam_mesh_generator = BeamMeshGenerator(self, ui, self.gestione_sezioni.section_manager, self.gestione_materiali)

        # --- INTEGRAZIONE CALCOLO ---

        btn_group_metodo_beam = QButtonGroup(self.parent)

        btn_group_metodo_beam.addButton(self.ui.btn_beam_continuo)
        btn_group_metodo_beam.addButton(self.ui.btn_beam_discontinuo)

        #comportamento esclusivo
        btn_group_metodo_beam.setExclusive(True)

        #Imposto i pulsanti come checkable
        self.ui.btn_beam_continuo.setCheckable(True)
        self.ui.btn_beam_discontinuo.setCheckable(True)

        QtCore.QTimer.singleShot(0, self.ui.btn_beam_continuo.click)

        self.beam_calcolo_dg = BeamCalcolo_dg(self.parent, self.ui, self.beam_mesh_generator)
        self.beam_calcolo = BeamCalcolo(self.parent, self.ui, self.beam_mesh_generator)

        # Collego una volta sola
        self.ui.beam_fem.clicked.connect(self.run_beam_analysis)

        # BTN GRUPPI MAIN
        btn_group_viste_beam = QButtonGroup(self.parent)

        #aggiungo pulsanti al gruppo
        btn_group_viste_beam.addButton(self.ui.btn_beam_3d)
        btn_group_viste_beam.addButton(self.ui.btn_beam_yz)
        btn_group_viste_beam.addButton(self.ui.btn_beam_xz)
        btn_group_viste_beam.addButton(self.ui.btn_beam_xy)

        #comportamento esclusivo
        btn_group_viste_beam.setExclusive(True)

        #Imposto i pulsanti come checkable
        self.ui.btn_beam_3d.setCheckable(True)
        self.ui.btn_beam_yz.setCheckable(True)
        self.ui.btn_beam_xz.setCheckable(True)
        self.ui.btn_beam_xy.setCheckable(True)

        QtCore.QTimer.singleShot(0, self.ui.btn_beam_3d.click)
        self.ui.progressBar_beam.setValue(0)

        # collega il pulsante che popola la combobox per il beam
        try:
            btn_main = getattr(self.ui, 'btn_main_beam', None)
            if btn_main is not None:
                btn_main.clicked.connect(self.populate_beam_combobox)
        except Exception:
            traceback.print_exc()

        # collega la combobox per tracciare la selezione
        try:
            combo = getattr(self.ui, 'combobox_beam_sezioni', None)
            if combo is not None and isinstance(combo, QComboBox):
                combo.currentIndexChanged.connect(self._on_beam_combo_changed)
        except Exception:
            traceback.print_exc()

    def run_beam_analysis(self):
        try:
            if self.ui.btn_beam_continuo.isChecked():
                self.beam_calcolo.start_fem_analysis()
            elif self.ui.btn_beam_discontinuo.isChecked():
                self.beam_calcolo_dg.start_fem_analysis()
            else:
                print("Nessun metodo selezionato.")
        except Exception:
            traceback.print_exc()

    def populate_beam_combobox(self):
        """
        Popola combobox_beam_sezioni mappando ciascun item all'indice reale
        della sezione (UserRole). Usa section_manager.sections se disponibile,
        altrimenti usa get_section_names_from_buttons() come fallback.
        """
        combo = getattr(self.ui, 'combobox_beam_sezioni', None)
        if combo is None:
            return
        try:
            combo.blockSignals(True)
            combo.clear()
            self._beam_section_map = []
            self.selected_section_index = None

            # Preferisci usare SectionManager.sections (ordine stabile)
            sm = None
            if hasattr(self.gestione_sezioni, 'section_manager'):
                sm = getattr(self.gestione_sezioni, 'section_manager')
            elif hasattr(self.gestione_sezioni, 'sections'):
                sm = self.gestione_sezioni

            if sm is not None and hasattr(sm, 'sections'):
                secs = getattr(sm, 'sections') or []
                if not secs:
                    combo.addItem("Nessuna sezione")
                    combo.setEnabled(False)
                    combo.blockSignals(False)
                    return

                combo.setEnabled(True)
                for section_index, sec in enumerate(secs):
                    # attempt to read name
                    name = None
                    try:
                        name = getattr(sec, 'name', None)
                        if name is None and isinstance(sec, dict):
                            name = sec.get('name')
                    except Exception:
                        name = None
                    if not name:
                        name = f"Sezione {section_index+1}"
                    combo.addItem(str(name))
                    # save section_index in itemData
                    combo.setItemData(combo.count() - 1, section_index, Qt.UserRole)
                    self._beam_section_map.append(section_index)

                combo.setCurrentIndex(0)
                # set selected_section_index
                if self._beam_section_map:
                    self.selected_section_index = self._beam_section_map[0]
                combo.blockSignals(False)
                return

            # Fallback: use names scraped from buttons / children
            names = self.get_section_names_from_buttons()
            if not names:
                combo.addItem("Nessuna sezione")
                combo.setEnabled(False)
                combo.blockSignals(False)
                return

            combo.setEnabled(True)
            for i, name in enumerate(names):
                combo.addItem(str(name))
                combo.setItemData(i, i, Qt.UserRole)
                self._beam_section_map.append(i)

            combo.setCurrentIndex(0)
            self.selected_section_index = self._beam_section_map[0] if self._beam_section_map else None
            combo.blockSignals(False)
        except Exception:
            traceback.print_exc()
            try:
                combo.blockSignals(False)
            except Exception:
                pass

    def _on_beam_combo_changed(self, comb_idx: int):
        """
        Aggiorna selected_section_index leggendo itemData (UserRole) o fallback sulla mappa.
        """
        combo = getattr(self.ui, 'combobox_beam_sezioni', None)
        if combo is None:
            self.selected_section_index = None
            return
        try:
            data = combo.itemData(comb_idx, Qt.UserRole)
            if data is None:
                # fallback: usa lista mappata
                if 0 <= comb_idx < len(self._beam_section_map):
                    self.selected_section_index = self._beam_section_map[comb_idx]
                else:
                    # ultima risorsa: usa comb_idx come indice
                    try:
                        self.selected_section_index = int(comb_idx)
                    except Exception:
                        self.selected_section_index = None
            else:
                try:
                    self.selected_section_index = int(data)
                except Exception:
                    self.selected_section_index = data
        except Exception:
            traceback.print_exc()
            try:
                self.selected_section_index = int(comb_idx)
            except Exception:
                self.selected_section_index = None