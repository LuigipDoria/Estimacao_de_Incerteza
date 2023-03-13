import matplotlib
import matplotlib.pyplot as plt

# Plota 1 grafico
def plot_graph(X, Y, title = "", xlabel = "", ylabel = "", line = True):
    fig1 = plt.figure()
    fig1.set_size_inches(25.5, 15.5)
    ax1 = fig1.add_subplot()
    fig1.subplots_adjust(top=0.85)
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    if line == True:
        ax1.plot(X,Y)
    else:
        ax1.scatter(X,Y)
    #pos1 = ax1.get_position()
    #ax1.set_position([pos1.x0, pos1.y0, pos1.width * 0.9, pos1.height])
    #ax1.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
    plt.grid()
    fig1.set_size_inches(8.5, 5.5)
    return fig1

# Plota mais de um grafico para o mesmo X
def plot_mutiple_graph(X, Y, n_graphs, title = "", xlabel = "", ylabel = "", line = True, label = "{}", multi_label=False):    
    if multi_label == True:
        label = label.replace(" ", "")
        label = label.split(",")
    fig1 = plt.figure()
    fig1.set_size_inches(25.5, 15.5)
    ax1 = fig1.add_subplot()
    fig1.subplots_adjust(top=0.85)
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    for i in range(n_graphs):
        if line == True:
            if multi_label == True:
                ax1.plot(X, Y[i], label=label[i])
            else:
                ax1.plot(X, Y[i], label=label.format(i+1))
        else:
            if multi_label == True:
                ax1.scatter(X, Y[i], label=label[i])
            else:
                ax1.scatter(X, Y[i], label=label.format(i+1))
    pos1 = ax1.get_position()
    ax1.set_position([pos1.x0, pos1.y0, pos1.width * 0.9, pos1.height])
    ax1.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
    plt.grid()
    fig1.set_size_inches(8.5, 5.5)
    return fig1
    
# Plota mais de um grafico para diferentes X e Y
def plot_mutiple_graph2(X, Y, n_graphs, title = "", xlabel = "", ylabel = "", line = True, label = "{}", multi_label=False):    
    if multi_label == True:
        label = label.replace(" ", "")
        label = label.split(",")
    fig1 = plt.figure()
    fig1.set_size_inches(25.5, 15.5)
    ax1 = fig1.add_subplot()
    fig1.subplots_adjust(top=0.85)
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    for i in range(n_graphs):
        if line == True:
            if multi_label == True:
                ax1.plot(X[i], Y[i], label=label[i])
            else:
                ax1.plot(X[i], Y[i], label=label.format(i+1))
        else:
            if multi_label == True:
                ax1.scatter(X[i], Y[i], label=label[i])
            else:
                ax1.scatter(X[i], Y[i], label=label.format(i+1))
    pos1 = ax1.get_position()
    ax1.set_position([pos1.x0, pos1.y0, pos1.width * 0.9, pos1.height])
    ax1.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
    plt.grid()
    fig1.set_size_inches(8.5, 5.5)
    return fig1