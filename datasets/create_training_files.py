from path import Path
import random

def main():
    root_path = "/home/md21local/visu_depth_haptic_data"
    root_path = Path(root_path)
    dirs = []
    dir_names = []

    for dir in root_path.dirs():
        dir_name = dir.name
        splitted_dir = dir_name.split("_")
        dirs.append(splitted_dir)
        dir_names.append(dir_name)
    
    materials = [dir[0] if len(dirs[0])==1 else dir[0][0] for dir in dirs]
    geometry = [dir[1] for dir in dirs]
    structure = [dir[2] for dir in dirs]
    color = [dir[3] for dir in dirs]
    position = [dir[4] for dir in dirs]

    samples = random.sample(dir_names, 60)
    rand_training = [s+'\n' for s in samples]
    rand_val = random.sample(dir_names, 1)

    mat_training, mat_val = [], []
    geo_training, geo_val = [], []
    struc_training, struc_val = [], []
    col_training, col_val = [], []
    pos_training, pos_val = [], []

    for i, (m, g, s, c, p) in enumerate(zip(materials, geometry, structure, color, position)):

        if m == 'E':
            mat_training.append(dir_names[i]+'\n')
        if m == 'D' and len(mat_val) < 1:
            mat_val.append(dir_names[i]+'\n')
        
        if g == 'P':
            geo_training.append(dir_names[i]+'\n')
        if g == 'S' and len(geo_val) < 1:
            geo_val.append(dir_names[i]+'\n')
        
        if s == 'S':
            struc_training.append(dir_names[i]+'\n')
        if s == 'D' and len(struc_val) < 1:
            struc_val.append(dir_names[i]+'\n')
        
        if c == 'P':
            col_training.append(dir_names[i]+'\n')
        if c == 'R' and len(col_val) < 1:
            col_val.append(dir_names[i]+'\n')
        
        if p in ['C', 'L1', 'R1']:
            pos_training.append(dir_names[i]+'\n')
        if p in ['R2', 'L2'] and len(pos_val) < 1:
            pos_val.append(dir_names[i]+'\n')

    file_dictionary = {'stiffness': [mat_training, mat_val],
                       'geometry': [geo_training, geo_val],
                       'structure': [struc_training, struc_val],
                       'color': [col_training, col_val],
                       'position': [pos_training, pos_val],
                       'random': [rand_training, rand_val]}
    
    for key in file_dictionary.keys():
        if key == "random":
            with open(root_path/'train.txt'.format(key), 'w') as file:
                file.writelines(file_dictionary[key][0])
            with open(root_path/'val.txt'.format(key), 'w') as file:
                file.writelines(file_dictionary[key][1])
        else:
            with open(root_path/'train_{}.txt'.format(key), 'w') as file:
                file.writelines(file_dictionary[key][0])
            with open(root_path/'val_{}.txt'.format(key), 'w') as file:
                file.writelines(file_dictionary[key][1])



if __name__ == "__main__":
    main()