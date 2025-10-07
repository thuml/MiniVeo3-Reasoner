from compare_traj import compare
from identify_traj import gen_traj
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_density(data, filename='density_plot.png', save_path='./', div=[10,20,30], title='Density Plot'):
    plt.figure(figsize=(10, 6))
    
    # define boundaries
    boundaries = div
    
    # plot the density plot as background
    sns.histplot(data, kde=True, stat="density", alpha=0.7)
    
    ranges = [(-np.inf, boundaries[0]), 
              (boundaries[0], boundaries[1]), 
              (boundaries[1], boundaries[2]), 
              (boundaries[2], np.inf)]
    
    plt.xlabel('maxdistance')
    plt.ylabel('density')
    plt.title(title)
    plt.grid(alpha=0.3)
    
    # add division lines
    for boundary in boundaries:
        plt.axvline(x=boundary, color='black', linestyle='--', alpha=0.7, linewidth=1)
        
    
    data=np.array(data)
    # calculate proportions and add text box
    cumulative_proportions = []
    proportions_text = f"Proportions: (total={len(data)})\n"
    
    for i, (start, end) in enumerate(ranges):
        if start == -np.inf:
            count = np.sum(data <= end)
        elif end == np.inf:
            count = np.sum(data > start)
        else:
            count = np.sum((data > start) & (data <= end))
        
        proportion = count / len(data)
        cumulative_proportions.append(proportion)
        
        if start == -np.inf:
            range_desc = f"≤{end}"
        elif end == np.inf:
            range_desc = f">{start}"
        else:
            range_desc = f"{start}-{end}"
        
        proportions_text += f"{range_desc}: {proportion:.1%}\n"
    
    # add text box
    plt.text(0.95, 0.95, proportions_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10, fontfamily='monospace')
    
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def eval_dis(dis, div):
    if dis > div[2]:
        evaluation = "Significantly Different"
    elif dis > div[1]:
        evaluation = "Slightly Similar"
    elif dis > div[0]:
        evaluation = "Moderately Similar"
    else:
        evaluation = "Highly Similar"
    return evaluation

def eval(dir, output_dir, epoch_name, div, threshold, omit=False):
    items = os.listdir(dir)
    similarity_groups = {
        'Highly Similar': [],
        'Moderately Similar': [],
        'Slightly Similar': [],
        'Significantly Different': []
    }
    max_distances = []
    
    png_items=[]
    
    for item in items:
        item_path = os.path.join(dir, item)
        if os.path.isfile(item_path) and item.lower().endswith('.png'):
            parts = item.split('_00.png')
            name = parts[0]
            # if os.path.isfile(os.path.join(dir, f"{name}_{epoch_name}_inference.mp4")) and os.path.isfile(os.path.join(dir, f"{name}.mp4")):
            if os.path.isfile(os.path.join(dir, f"{name}_inference.mp4")) and os.path.isfile(os.path.join(dir, f"{name}.mp4")):
                png_items.append(item)
    
    total_num=len(png_items)
    
    if total_num==0:
        return
    from tqdm import tqdm
    
    # colored output function
    def colored_similarity_output(filename, similarity_type):
        colors = {
            'Moderately Similar': '\033[92m',  # green
            'Slightly Similar': '\033[93m',     # yellow
            'Significantly Different': '\033[91m'  # red
        }
        color_code = colors.get(similarity_type, '')
        reset_code = '\033[0m'
        
        if similarity_type in colors:
            return f"{color_code}{filename}: {similarity_type}{reset_code}"
        return None

    # tqdm loop
    for item in tqdm(png_items, desc=f"Processing {dir}", total=total_num, 
                    bar_format="{l_bar}%s{bar}%s{r_bar}" % ('\033[94m', '\033[0m')):
        parts = item.split('_00.png')
        name = parts[0]
        file1 = f"{name}.mp4"
        # file2 = f"{name}_{epoch_name}_inference.mp4"
        file2 = f"{name}_inference.mp4"
        

        expert, student = gen_traj(f"{dir}/{file1}", f"{dir}/{file2}")
        max_index, max_distance, dis, _1, _2 = compare(expert, student)
        first_diff = 5000
        for i in range(len(dis)):
            if dis[i] > threshold:
                first_diff = i
                break
        metrics = {
            "max_distance": max_distance,
            "first_diff": first_diff,
            "PR": first_diff/len(dis),
            "similarity_evaluation": eval_dis(max_distance, div)
        }
        max_distances.append(max_distance)
        similarity_groups[metrics['similarity_evaluation']].append((file2, metrics))
        
        # output results (omit Highly Similar)
        colored_output = colored_similarity_output(file2, metrics['similarity_evaluation'])
        if colored_output:
            tqdm.write(colored_output)

    plot_density(max_distances,"distances_plot",output_dir,div,title=f"{dir} {epoch_name}")
    
    for evaluation, files in similarity_groups.items():
        filename = f"{evaluation.replace(' ', '_').lower()}_files.txt"
        filename = f"{output_dir}/{filename}"
        
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"{evaluation} Files\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total: {len(files)} files.\n\n")
            
            for file in files:
                f.write(f"{file[0]}: dist={file[1]['max_distance']}, PR={file[1]['PR']}\n")
        if not omit:
            print(f"Generated file: {filename}")
    
import sys
import argparse

def main():
    # default parameters
    default_dir = "./"
    default_output_dir = "./result"
    # default_epoch_name = "epoch-0"
    default_div = [10, 20, 30]
    default_threshold = 20
    
    # argparse settings
    parser = argparse.ArgumentParser(description='Evaluate video reasoning results.')
    parser.add_argument('--input-dir', '-i', type=str, default=default_dir, help='Evaluation data input directory.')
    parser.add_argument('--output-dir', '-o', type=str, default=default_output_dir, help='Evaluation result output directory')
    # parser.add_argument('--epoch-name', '-e', type=str, default=default_epoch_name, help='Epoch name in filename. Your output should be like xxx_{epoch_name}_inference.mp4')
    parser.add_argument('--div', '-d', type=str, default=','.join(map(str, default_div)), help='Division points for categories, e.g., "10,20,30"')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode, suppress output')
    parser.add_argument('--threshold', '-t', type=int, default=default_threshold, help='The threshold pixel distance to determined to be wrong, default=20.')

    args = parser.parse_args()
    dir_path = args.input_dir
    output_dir = args.output_dir
    epoch_name ="miniveo3-reasoner-maze"# args.epoch_name
    div = list(map(int, args.div.split(','))) if args.div else default_div
    threshold = args.threshold
    omit = args.quiet
    
    if not omit:
        print(f"Evaluation data input directory: {dir_path}")
        print(f"Evaluation result output directory: {output_dir}")
        # print(f"Epoch name: {epoch_name}")
        print(f"Division points: {div}")
        print(f"Threshold: {threshold}")

    eval(dir_path, output_dir, epoch_name, div, threshold, omit)

if __name__ == "__main__":
    main() 