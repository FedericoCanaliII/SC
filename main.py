import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QIcon, QColor, QFont
from PyQt5.QtWidgets import QApplication, QButtonGroup, QVBoxLayout, QTextEdit

from interfaccia.main_interfaccia import Ui_MainWindow
from sezione.gestione_sezione import GestioneSezioni 
from output.gestione_output import GestioneOutput
from momentocurvatura.gestione_momentocurvatura import GestioneMomentocurvatura
from beam.gestione_beam import GestioneBeam

# --- CLASSE PER CATTURARE L'OUTPUT DEL TERMINALE ---
class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self): 
        super(MainWindow, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # LOGO APP
        self.setWindowTitle("SectionCHECK")
        # Nota: Assicurati che il percorso esista, altrimenti l'icona non si carica
        icon_path = "C:\\Users\\canal\\Desktop\\SC\\interfaccia\\icone\\logo.ico"
        self.setWindowIcon(QIcon(icon_path))

        # SCHERMO INTERO
        self.showMaximized()

        # ---------------------------------------------------------
        # INTEGRAZIONE TERMINALE IN widget_terminale
        # ---------------------------------------------------------
        # 1. Creiamo un layout per il widget contenitore (se non ne ha già uno nel .ui)
        if self.ui.widget_terminale.layout() is None:
            self.terminal_layout = QVBoxLayout(self.ui.widget_terminale)
            self.terminal_layout.setContentsMargins(0, 0, 0, 0)
        else:
            self.terminal_layout = self.ui.widget_terminale.layout()

        # widget_terminale_2
        if self.ui.widget_terminale_2.layout() is None:
            self.terminal_layout_2 = QVBoxLayout(self.ui.widget_terminale_2)
            self.terminal_layout_2.setContentsMargins(0, 0, 0, 0)
        else:
            self.terminal_layout_2 = self.ui.widget_terminale_2.layout()

        # 2. Creiamo l'area di testo che fungerà da terminale (Terminale 1)
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)  # L'utente non può scriverci dentro
        
        # Terminale 2
        self.console_output_2 = QTextEdit()
        self.console_output_2.setReadOnly(True)

        # 3. Stile "Hacker/Terminale" (Sfondo nero, testo verde o bianco, font monospaziato)
        style_sheet_terminal = """
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
                border: none;
            }
        """
        self.console_output.setStyleSheet(style_sheet_terminal)
        self.console_output_2.setStyleSheet(style_sheet_terminal)
        
        # 4. Aggiungiamo l'area di testo al widget_terminale
        self.terminal_layout.addWidget(self.console_output)
        self.terminal_layout_2.addWidget(self.console_output_2)

        # 5. Redirezione di sys.stdout e sys.stderr
        self.sys_stdout_backup = sys.stdout  # Salviamo l'originale per sicurezza
        self.sys_stderr_backup = sys.stderr

        self.output_stream = EmittingStream()
        self.output_stream.textWritten.connect(self.append_terminal_text)

        sys.stdout = self.output_stream
        sys.stderr = self.output_stream # Cattura anche gli errori
        
        print(">> Terminale Inizializzato correttamente...")
        print(">> Benvenuto in SectionCHECK")
        # ---------------------------------------------------------

        # PULSANTI AUTOMATICI
        QtCore.QTimer.singleShot(0, self.ui.btn_main_sc.click)

        # GESTIONE DELLE PAGINE
        self.ui.btn_main_sc.clicked.connect(lambda: self.ui.stackedWidget_main.setCurrentIndex(0))
        self.ui.btn_main_materiali.clicked.connect(lambda: self.ui.stackedWidget_main.setCurrentIndex(1))
        self.ui.btn_main_input.clicked.connect(lambda: self.ui.stackedWidget_main.setCurrentIndex(2))
        self.ui.btn_main_output.clicked.connect(lambda: self.ui.stackedWidget_main.setCurrentIndex(3))
        self.ui.btn_main_momentocurvatura.clicked.connect(lambda: self.ui.stackedWidget_main.setCurrentIndex(4))
        self.ui.btn_main_beam.clicked.connect(lambda: self.ui.stackedWidget_main.setCurrentIndex(5))

        # BTN GRUPPI MAIN
        btn_group_main = QButtonGroup(self)

        #aggiungo pulsanti al gruppo
        btn_group_main.addButton(self.ui.btn_main_sc)
        btn_group_main.addButton(self.ui.btn_main_materiali)
        btn_group_main.addButton(self.ui.btn_main_input)
        btn_group_main.addButton(self.ui.btn_main_output)
        btn_group_main.addButton(self.ui.btn_main_momentocurvatura)
        btn_group_main.addButton(self.ui.btn_main_beam)

        #comportamento esclusivo
        btn_group_main.setExclusive(True)

        #Imposto i pulsanti come checkable
        self.ui.btn_main_sc.setCheckable(True)
        self.ui.btn_main_materiali.setCheckable(True)
        self.ui.btn_main_input.setCheckable(True)
        self.ui.btn_main_output.setCheckable(True)
        self.ui.btn_main_momentocurvatura.setCheckable(True)
        self.ui.btn_main_beam.setCheckable(True)

        # GESTIONE SEZIONI
        self.sezioni = GestioneSezioni(self.ui)
        self.output = GestioneOutput(self.ui, self.sezioni, self.sezioni.gestione_materiali)
        self.momentocurvatura = GestioneMomentocurvatura(self.ui, self.sezioni, self.sezioni.gestione_materiali)
        self.beam = GestioneBeam(self, self.ui, self.sezioni, self.sezioni.gestione_materiali)

        # TOOLTIP
        self.ui.btn_main_sc.setToolTip("SectionCHECK")
        self.ui.btn_main_lingua.setToolTip("Lingua")
        self.ui.btn_main_colore.setToolTip("Tema")
        self.ui.btn_main_salva.setToolTip("Salva")
        self.ui.btn_main_carica.setToolTip("Carica")
        self.ui.btn_main_stampa.setToolTip("Stampa")

        self.ui.progressBar_verifica.setValue(0)
        self.ui.progressBar_verifica_MC.setValue(0)

        # COLLEGAMENTO TASTO STAMPA
        self.ui.btn_main_stampa.clicked.connect(self.salva_screenshot)

    def append_terminal_text(self, text):
        """Funzione chiamata ogni volta che c'è un print()"""
        # --- TERMINALE 1 ---
        # Sposta il cursore alla fine per evitare di scrivere in mezzo
        cursor = self.console_output.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.console_output.setTextCursor(cursor)
        
        # Inserisce il testo
        self.console_output.insertPlainText(text)
        
        # Scorrimento automatico in basso
        sb = self.console_output.verticalScrollBar()
        sb.setValue(sb.maximum())

        # --- TERMINALE 2 ---
        cursor2 = self.console_output_2.textCursor()
        cursor2.movePosition(QtGui.QTextCursor.End)
        self.console_output_2.setTextCursor(cursor2)
        
        self.console_output_2.insertPlainText(text)
        
        sb2 = self.console_output_2.verticalScrollBar()
        sb2.setValue(sb2.maximum())

    def closeEvent(self, event):
        """Ripristina stdout alla chiusura per evitare errori in console IDE"""
        sys.stdout = self.sys_stdout_backup
        sys.stderr = self.sys_stderr_backup
        super().closeEvent(event)

    def salva_screenshot(self):
        """Cattura uno screenshot fedele della finestra usando lo schermo."""
        opzioni = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 
            "Salva Screenshot", 
            "Screenshot_SectionCHECK.png", 
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)", 
            options=opzioni
        )

        if file_path:
            # Otteniamo lo schermo su cui si trova la finestra
            screen = QtWidgets.QApplication.primaryScreen()
            if screen:
                # Cattura esattamente l'area occupata dalla finestra (compresi i widget OpenGL)
                # winId() permette di catturare il contenuto renderizzato dalla GPU
                screenshot = screen.grabWindow(self.winId())
                
                successo = screenshot.save(file_path)

                if successo:
                    print(f">> Screenshot salvato (metodo Screen): {file_path}")
                else:
                    print(">> Errore nel salvataggio.")
             
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())