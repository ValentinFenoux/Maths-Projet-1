# Imports
import matplotlib.pyplot as plt
import autograd
from autograd import numpy as np
# Amorce

def find_seed(g, c=0, eps=2**(-26)) :
    if (c >= g(0,0) and c <= g(0,1)) or (c <= g(0,0) and c >= g(0,1)) :
        #-----------------------------#
        def f(y) :
            return(g(0,y) - c)
            
        def dichotomie(f,a,b) :
            while abs(a - b) > eps :
                c = (a+b)/2
                if f(c)*f(a) > 0 :
                    a = c
                else :
                    b = c
            return(c)
        #------------------------------#
        t = dichotomie(f,0,1)
        return(t)
    else :
        return(None)

def norme(X):
    return (X[0]**2 + X[1]**2)**(1/2)

def simple_contour(f, c=0.0, delta=0.01) :
    X = [] ; Y = []
    # Définition du gradient
    def grad_f(x,y) :
        g = autograd.grad
        grad = np.array([g(f,0)(x,y),g(f,1)(x,y)])
        if norme(grad) == 0:
            return grad
        else:
            return(grad/norme(grad))  # On retourne un gradient normé
    # Recherche de t sur la frontière
    y = find_seed(f,c) ; x = 0.0
    print(y)
    if y != None :
        X.append(0) ; Y.append(y)
    else :
        return(X,Y)     # Cas où c n'est pas sur la frontière
    gf = grad_f(x,y)
    if norme(gf) == 0:    #si le gradient s'annule c'est qu'on est sur un extremum donc on s'arrête là
        return (X,Y)
    elif gf[1] >= 0 :   #on choisit le bon vecteur orthogonal au gradient pour rester dans la cellule
        E = 1
    else :
        E = -1
    # Recherche de la ligne de niveau
    while x >= 0 and x <= 1 and y >= 0 and y <= 1 :
        gf = grad_f(x,y)
        if norme(gf) == 0:
            return X,Y
        x += E*gf[1]*delta ; y -= E*gf[0]*delta
        X.append(x) ; Y.append(y)
    
    X.pop() ; Y.pop() #le dernier point est hors du cadre
    return(X,Y)
#--------------------Contour complexe--------------------#
#----------Seed pour le contour complexe----------#
def seed_complexe(g, x0, x1, y0, y1, c=0.0, eps=2**(-26)) :
    T = []
    #---------Fonction pour la recherche de t-----------#
    def dichotomie(f,a,b) :
            while abs(a - b) > eps :
                c = (a+b)/2
                if f(c)*f(a) > 0 :
                    a = c
                else :
                    b = c
            return(c)
    #---------On cherche le seed sur les 4 côtés du carré-----------#
    if (c >= g(x0,y0) and c <= g(x0,y1)) or (c <= g(x0,y0) and c >= g(x0,y1)) :
        def f1(y) :
            return(g(x0,y) - c)
        t = dichotomie(f1,y0,y1)
        T.append([x0,t])
    #--------------------#   
    if (c >= g(x0,y0) and c <= g(x1,y0)) or (c <= g(x0,y0) and c >= g(x1,y0)) :
        def f2(x) :
            return(g(x,y0) - c)
        t = dichotomie(f2,x0,x1)
        T.append([t,y0])
    #--------------------#   
    if (c >= g(x1,y1) and c <= g(x1,y0)) or (c <= g(x1,y1) and c >= g(x1,y0)) :
        def f3(y) :
            return(g(x1,y) - c)
        t = dichotomie(f3,y0,y1)
        T.append([x1,t])
    #--------------------#
    if (c >= g(x1,y1) and c <= g(x0,y1)) or (c <= g(x1,y1) and c >= g(x0,y1)) :
        def f4(x) :
            return(g(x,y1) - c)
        t = dichotomie(f4,x0,x1)
        T.append([t,y1])
    return(T)
    
#-------------------------------#
def cplx_contour(f, x0, x1, y0, y1, c=0.0, delta=0.01) :
    X = [] ; Y = []
    # Définition du gradient
    def grad_f(x,y) :
        g = autograd.grad
        grad = np.array([g(f,0)(x,y),g(f,1)(x,y)])
        if norme(grad) == 0:
            return grad
        else:
            return(grad/norme(grad))  # On retourne un gradient normé
    # Recherche de t sur la frontière
    T = seed_complexe(f, x0, x1, y0, y1, c)
    for x,y in T :
        X.append(x) ; Y.append(y)
        gf = grad_f(x,y)
        if norme(gf) == 0:
            return X,Y
        # Il y a différentes conditions pour avoir un vecteur 
        # orthogonal au gradient dirigé vers l'intérieur du cadre
        if x == x0 :
            if gf[1] >= 0 :
                E = 1
            else :
                E = -1
        elif x == x1 :
            if gf[1] >= 0 :
                E = -1
            else :
                E = 1
        elif y == y0 :
            if gf[0] >= 0 :
                E = 1
            else :
                E = -1 
        else : 
            if gf[0] >= 0 :
                E = -1
            else :
                E = 1
        # Recherche de la ligne de niveau
        while x >= x0 and x <= x1 and y >= y0 and y <= y1 :
            gf = grad_f(x,y)
            x += E*gf[1]*delta ; y -= E*gf[0]*delta
            X.append(x) ; Y.append(y)
        X.pop() ; Y.pop() #le dernier point est hors du cadre
    return(X,Y)

    
def contour(f, c=0.0, xc=[0.0,1.0], yc=[0.0,1.0], delta=0.01) :
    xs = [] ; ys = []
    for i in range(len(xc)-1) :
        for j in range(len(yc)-1) :
            x,y = cplx_contour(f, xc[i], xc[i+1],yc[j],yc[j+1],c,delta)
            xs.append(x) ; ys.append(y)
    return(xs,ys)
        
    
#-----------------------------------------#
def f(x,y) :
    return(2*(np.exp(-x**2-y**2) - np.exp(-(x-1)**2-(y-1)**2)))

def g(x,y) :
    return(x**2 + y**2)


    
C = [0.0,0.5,1.0,1.5,2.0]
xc = np.linspace(-2,2,40)
yc = np.linspace(-2,2,40)

for c in C :
    xs,ys = contour(g,c,xc,yc)
    for x,y in zip(xs,ys) :
        plt.plot(x,y,'b') 

plt.xlim(-2,2)
plt.ylim(-2,2)
plt.show()


                  
    
    
        