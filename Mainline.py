import os
import sys

# set working directory so relative paths work
def set_working_directory():
    script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    if getattr(sys, 'frozen', False):
        # Running as executable
        project_path = os.path.abspath(os.path.join(script_path, '..'))
    else:
        # Running from source directory
        project_path = script_path
    os.chdir(project_path)

set_working_directory()

from PyQt6 import QtWidgets
import GUI
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    ui = GUI.Ui()
    ui.show()
    app.exec()