 #######################################
# line_of_best_fit.py
# Erica Lastufka 18/9/17 

#Description: Get the line of best fit to the data, given certain conditions
#######################################

import numpy as np

def line_of_best_fit(datax,datay,loglog= True,minx=False,maxx=False,miny=False, maxy=False,minratio=False,maxratio=False,plot=False):
    #if list convert to np.array
    if type(datax) == list: datax=np.array(datax)
    if type(datay) == list: datay=np.array(datay)
    if not minx and not maxx: fdatax=datax
    if not miny and not maxy: fdatay=datay
        
    for x in datax:
        if minx and x > minx:
            fdatax.append(x)
        if maxx and x < maxx:
            fdatax.append(x)
    for y in datay:
        if miny and y > miny:
            fdatayx.append(y)
        if maxy and y < maxy:
            fdatay.append(y)
    if type(fdatax) == list: fdatax=np.array(fdatax)
    if type(fdatay) == list: fdatay=np.array(fdatay)
            
    if loglog:
        fdatax=np.log10(fdatax)
        fdatay=np.log10(fdatay)
        
    #ratio=datay/datax deal with this later

    line=polyfit(fdatax,fdatay,1) 

    if plot:
        fig,ax=plt.subplots()
        ax.scatter(datax,datay)
        ax.plot(line)
        fig.show()
    
    return line
    
