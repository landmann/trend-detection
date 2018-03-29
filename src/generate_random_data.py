import math
import random
import matplotlib
matplotlib.use('Agg')

import matplotlib.style as style
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
#import pandas as pd
import numpy as np
import PIL
from scipy.signal import bspline
from scipy.interpolate import splrep, splev
from scipy import stats
from IPython.display import Image, display, clear_output
import io


DATA_PATH = '../data/'

##############################################################################################

STYLES = style.available 

#FUNCTIONS = (linear_func, linear_neg_func, log_func, neg_log_func, polynomial, exponential)
OFFSETS = (-10, 10)

NUM_TRENDS = [1, 1]

# CHARTTYPES = ('line', 'bar', 'scatter')
LABELSIZES = (1, 5)
#FIG_WIDTH = (5, 25)
FIG_WIDTH = (5, 25)
ASP_RATIO = (.5, 2)
DPI=50

CHART_RANGE = (-10, 10)

PROB_SOLID_LINE = .5
LINESTYLES = ('-', ':', '-.', '--')
LINEWIDTHS = [.5, 1.5] # as a fraction of the number of points

NUM_POINTS = (5, 50)
NUM_POINTS_INTERP = (0, 5000)
NOISE_DEV = (0, 5)

TITLE_PROB = .75
TITLE_LENGTH = (5, 100)
TITLE_POSITIONS = ['left', 'right', 'center']
TITLE_FONTSIZE = (10, 20)
TITLE_FONTWEIGHT = (1, 10)

##############################################################################################

def prep_data(func, noise_mean, noise_dev, num_points, smooth, num_points_interp):
    # args include 'func', 'name', and 'linetype'.
    noise = np.random.normal(noise_mean, noise_dev, num_points)
    x = np.array([i for i in range(num_points)])
    
    #yvals = func(x+noise)
    yvals = func(x) + noise
    
    if smooth:
        x_range = np.linspace(x[0],x[-1], num_points_interp)
        yvals = splev(x_range, splrep(x, yvals))
        x = x_range
        
    #colname = line.get('name', None)
    #data = pd.DataFrame(yvals, index=x,columns=colname)
    data = [x, yvals]
    return data

def plot_line(func, noise_mean, noise_dev, num_points, smooth, num_points_interp, normalize=False, **styles): 
    x, y = prep_data(func, noise_mean, noise_dev, num_points, smooth, num_points_interp)
    
    if normalize: 
        # Numbers are between 1 and 3
        max_val = max(np.abs(y))
        if max_val > 3: 
            factor = max_val/random.uniform(1, 3)
            y = y/factor
        
    plt.plot(x, y, **styles)
    return x, y

def plot_multi_lines(lines, normalize=False, **graph_style):
    plt.clf()
    # first set up the chart 
    graph = plt.figure(figsize=graph_style['figsize'])
    
    plt.style.use(graph_style['style'])
    
    if graph_style.get('use_title', True): 
        title_args = {
            's': graph_style.get('title', 'My Title'),
            'loc': graph_style.get('title_location', 'center'),
        }
        if graph_style.get('title_font_dict', False): 
            title_args['fontdict'] = graph_style['fontdict']
        plt.title(**title_args)
    
    min_x = math.inf
    max_x = -math.inf
    styles_included = []
    for i, line in enumerate(lines):
        x, y = plot_line(normalize=normalize, **line)
        slope, intercept, r_val, p_val, stderr = stats.linregress(x, y)
        styles_included.append({
            'slope': slope,
            'intercept': intercept,
            'r_val': r_val,
            'p_val': p_val,
            'stderr': stderr
        })
        min_x = min(x[0], min_x)
        max_x = max(x[-1], max_x)
        
    axes = graph.axes[0]
    axes.tick_params(axis = 'both', which = 'major', labelsize = graph_style['labelsize'])
    axes.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .7) #Bolding horizontal line at x
    axes.xaxis.label.set_visible(False)
    axes.set_xlim(left =min_x, right=max_x)
    axes.set_ylim(top=CHART_RANGE[1] + NOISE_DEV[1], bottom=CHART_RANGE[0] - NOISE_DEV[1])
    if normalize: 
        axes.set_ylim(bottom=-3.1, top=3.1)
    
    graph.legend_ = None

    return styles_included

def random_line(func_type, width=10, verbose=False, **kwargs): 
    #offset = [0, 0]
    #offset = kwargs.get('offset', [random.uniform(OFFSETS[0], OFFSETS[1]) for i in range(2)]) # len-2 list
    #func = lambda x: base_func(x, offset=offset)
    
    numpoints = kwargs.get('num_points', random.randrange(NUM_POINTS[0], NUM_POINTS[1] + 1))
    #name, function = random_func(numpoints)
    #func, noise_dev, func_params = function
    func, func_params = rand_graph_in_range(numpoints, func_type=func_type)
    noise_dev = random.uniform(NOISE_DEV[0], NOISE_DEV[1])
    if func_type == 'sinusoid': 
        noise_dev = noise_dev/2
    
    linewidth = kwargs.get('linewidth', None)
    if not linewidth: 
        linewidth = .5*width
        linewidth = linewidth*random.uniform(LINEWIDTHS[0], LINEWIDTHS[1])
        linewidth = int(round(linewidth))
    
    use_straight_line = random.random() < PROB_SOLID_LINE
    linestyle = '-' if use_straight_line else random.choice(LINESTYLES)
    
    vals = {
        'func': func,
        'num_points': numpoints,
        'num_points_interp': kwargs.get('num_points_interp', random.randrange(NUM_POINTS_INTERP[0], NUM_POINTS_INTERP[1] + 1)),
        'noise_mean': 0, #I think any number is just equivalent to adding an offset. 
        'noise_dev': kwargs.get('noise_dev', noise_dev), 
        'smooth': kwargs.get('smooth', random.choice([True, False])),
        'linestyle': kwargs.get('linestyle', linestyle),
        'linewidth': linewidth
    }
    if verbose: 
        print("Generated line graph with:\n"
             "\tFunction: {} with params {}\n"
              "\t Num Points {}\n"
              "\t Noise Dev {}\n" 
              "\t Linestyle {}\n"
              "\t Linewidth {}\n".format(func.__name__, 
                                         func_params, 
                                         vals['num_points'], 
                                         vals['noise_dev'], 
                                         vals['linestyle'], 
                                         vals['linewidth'])         
        )
    func_params['numpoints'] = numpoints
    return vals, func_params

def random_line_in_range(numpoints): 
    l_endpoint = random.uniform(CHART_RANGE[0], CHART_RANGE[1])
    r_endpoint = random.uniform(-10, 10)
    #figure out the line function that connects these two points
    m = (r_endpoint - l_endpoint)/(numpoints)
    # Use the point (0, l_endpoint)
    f = lambda x: m*x + l_endpoint
    params = {
        'start': l_endpoint,
        'end': r_endpoint,
        'm': m
    }
    return f, params

def random_sinusoid_in_range(numpoints): 
    period = numpoints-1
    phi = 0 if .5 < random.random() else math.pi/2 
    amplitude = random.uniform(CHART_RANGE[1]/5, CHART_RANGE[1])
    sign = 1 if .5 < random.random() else -1 
    offset = random.uniform(0, CHART_RANGE[1] - amplitude)
    f = lambda x: sign*amplitude*np.cos(2*math.pi/period*x - phi) + offset 
    params = {
        'phi': phi,
        'numpoints': numpoints,
        'amplitude': sign*amplitude,
        'offset': offset
    }
    return f, params

def random_exp_in_range(numpoints): 
    flip = 1 if .5 < random.random() else -1
    flip_horiz = 1 if .5 < random.random() else -1
    exp = random.uniform(5, 10)
    #simple_f = lambda x: x**exp
    simple_f = lambda x: np.exp(1.5*x)
    func_max = simple_f(numpoints-1)
    # scale the whole function so that it's las tpoint takes up 1/2 to 1 times the graph rnage 
    g_range = CHART_RANGE[1] - CHART_RANGE[0]
    func_range = g_range*random.uniform(.5, 1)
    # max_val/factor = func_range ==> factor = func_range/max_val 
    unsigned_f = lambda x: simple_f(x)*func_range/func_max
    if flip < 0: 
        signed_f = lambda x: unsigned_f(numpoints-1) - unsigned_f(x)
    else: 
        signed_f = unsigned_f
    if flip_horiz < 0: 
        flipped_f = lambda x: signed_f(numpoints-x)
    else: 
        flipped_f = signed_f
        
    offset = random.uniform(CHART_RANGE[0], CHART_RANGE[1] - func_range)
    f = lambda x: flipped_f(x) + offset
    params = {
        #'exp': exp,
        'flip_v': flip,
        'flip_h': flip_horiz,
        'offset': offset,
        'func_range': func_range
    }
    return f, params

def rand_graph_in_range(numpoints, func_type='line'): 
    if func_type == 'line': 
        return random_line_in_range(numpoints)
    elif func_type == 'sinusoid': 
        return random_sinusoid_in_range(numpoints)
    elif func_type == 'exponential':
        return random_exp_in_range(numpoints)
    else: 
        return None

def random_title(verbose=False, **kwargs): 
    use_title = random.random() < TITLE_PROB
    if not use_title: 
        if verbose: 
            print("Not using a title")
        return {'use_title': False}
    letter_options = list("abcdefghijklmnopqrstuvxyzABCDEFGHIJKLMNOPWRSTUVXYZ ,!:.")
    title_len = random.randrange(TITLE_LENGTH[0], TITLE_LENGTH[1] + 1)
    title_args = {
        'title': kwargs.get('title', "".join([random.choice(letter_options) for i in range(title_len)])),
        'loc': kwargs.get('title_location', random.choice(TITLE_POSITIONS)),
        'fontdict': {
            'fontsize': kwargs.get('fontsize', random.randrange(TITLE_FONTSIZE[0], TITLE_FONTSIZE[1] + 1)),
            'fontweight': random.randrange(TITLE_FONTWEIGHT[0] + TITLE_FONTWEIGHT[1] + 1)
        }
    }
    if verbose: 
        print("Using a title with settings:\n" 
              "\tfontsize: {}\n"
              "\tfontweight: {}\n"
              "\tlocation: {}\n"
              "\tlength: {}\n".format(title_args['fontdict']['fontsize'], 
                                      title_args['fontdict']['fontweight'], 
                                      title_args['loc'], title_len)
             )
    return title_args

def random_line_graph(func_type='line', **kwargs): 
    # now we have our lines determined. 
    if 'style' in kwargs: 
        style = kwargs.get('style')
    else: 
        style = random.choice(STYLES)
        
    figsize = kwargs.get('figsize', 0)
    if not figsize: 
        w = random.randrange(FIG_WIDTH[0], FIG_WIDTH[1] + 1)
        h = random.uniform(ASP_RATIO[0], ASP_RATIO[1])*w
        figsize = (h, w)
    
    #print("Plot with width %d and height %d\n" % (w, h))
        
    # fontsize needs to be a function of the width of the graph
    fontsize = int(round(1/10*w*random.uniform(LABELSIZES[0], LABELSIZES[1])))
    
    graph_styles = {
        'labelsize': kwargs.get('labelsize', 5*fontsize),
        'figsize': figsize, # random w and h
        'style': style
    }
    title_args = random_title(fontsize=fontsize)
    graph_styles = dict(**graph_styles, **title_args)
    
    lines = kwargs.get('lines', [])
    num_lines = kwargs.get('num_lines', random.randrange(NUM_TRENDS[0], NUM_TRENDS[1] + 1))
    params = []
    if len(lines) < num_lines: 
        num_points = random.randrange(NUM_POINTS[0], NUM_POINTS[1] + 1)
        for i in range(num_lines - len(lines)): 
            line, p = random_line(func_type=func_type, width=w, num_points=num_points)
            lines.append(line)
            params.append(p)
            #vals, name = random_line(width=w, num_points=num_points)
            # calculate t-val of vals 
            #vals = prep_data(vals)
            
            #labels.append(name)
            #lines.append(vals)
    
    lines_added = plot_multi_lines(lines, **graph_styles)
    return lines_added, params


def get_label(lines_data, params_data, func_type): 
    STABLE = 0
    INCREASING = 1
    DECREASING = -1
    if func_type == 'line': 
        p_val = lines_data['p_val']
        reg_slope = lines_data['slope']
        numpoints = params_data['numpoints']
        change = reg_slope*numpoints
        fraction_of_y = change/(CHART_RANGE[1]-CHART_RANGE[0])
        label = None
        if p_val > .05 or abs(fraction_of_y) < .01: 
            return STABLE
        else: 
            if reg_slope < 0: 

                return DECREASING
            else: 
                return INCREASING
    elif func_type == 'sinusoid': 
        return STABLE
    elif func_type == 'exponential': 
        flip_h = params_data['flip_h']
        flip_v = params_data['flip_v']
        direction = flip_h*flip_v
        if direction > 0: 
            return INCREASING
        else: 
            return DECREASING
    else: 
        return None

def make_data(set_type, num_img): 
    with open(DATA_PATH + set_type + '_labels.txt', 'w') as outfile: 
        for i in range(0, num_img): 
            fig_path = DATA_PATH + set_type + '/' + str(i) 
            lines_data, params_data = random_line_graph()
            label = get_label(lines_data[0], params_data[0])
            outfile.write("{}.png {}\n".format(i, label))
            savefig(fig_path, dpi=DPI)
            resize_image(fig_path + '.png', show=False)
            plt.close()  

def save_image(path, size=(224, 224), show=False): 
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = PIL.Image.open(buf)
    im = im.resize(size)
    im.save(path)
    buf.close()
    im.close()
    if show: 
        img = Image(path)
        display(img)
    
        
def add_data(start, end, set_type, func_type): 
    with open(DATA_PATH + set_type + '_labels_' + func_type + '.txt', 'a') as outfile: 
        for i in range(start, end): 
            while True: # keep trying to do this
                try:
                    fig_path = DATA_PATH + set_type + '/' + func_type + '/' + str(i) 
                    lines_data, params_data = random_line_graph(func_type=func_type)
                    label = get_label(lines_data[0], params_data[0], func_type=func_type)
                    outfile.write("%s/%d.png %d\n" % (func_type, i, label))
                    #savefig(fig_path, dpi=300)
                    #resize_image(fig_path + '.png', show=False)
                    save_image(fig_path + '.png', show=False)
                    # re-read the figure, downsize 
                    plt.close()
                    break
                except Exception as e:
                    print("ERROR")
                    print(e)
                    continue

add_data(0, 1250, 'val', 'exponential')
