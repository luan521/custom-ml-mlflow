import os
import logging
def find_project_path():
    """
    Find local path to project. OBS: you must be inside a git project directory
    
    Returns:
        (str) local path to project
    """
    
    arg_abspath = ''
    path_project = os.path.abspath(arg_abspath)
    while '.git' not in os.listdir(
                                   path_project
                                  ):
        arg_abspath = arg_abspath + '../'
        path_project = os.path.abspath(arg_abspath)
        if path_project == '/':
            logging.error('ERROR: Could not find project path.')
            return
        
    return path_project