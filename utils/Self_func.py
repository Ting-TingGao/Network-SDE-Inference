"""Self library
   author: Ting-Ting Gao"""

import pandas as pd
import numpy as np
import itertools as it

def elementary_functions_name(dimensionList,order):
    Combination_func = list(it.combinations_with_replacement(dimensionList,order))
    Num_of_func = len(Combination_func)
    Name_of_func = []
    for i in range(0,Num_of_func):
        tmp = "".join(Combination_func[i])
        Name_of_func.append(tmp)
    return Num_of_func, Name_of_func

def self_ElementaryFunctions_Matrix(TimeSeries, dim, selfPolyOrder, PolynomialIndex = True, TrigonometricIndex = True, \
    ExponentialIndex = True, FractionalIndex = True, ActivationIndex = True):
    
    ElementaryMatrix = pd.DataFrame()
    if PolynomialIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix,Polynomial_functions(TimeSeries, dim, selfPolyOrder)],axis=1)
    if TrigonometricIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Trigonometric(TimeSeries, dim,Sin = True, Cos = True, Tan = True)],axis=1)
    if ExponentialIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Exponential(TimeSeries, dim, expomential = True)],axis=1)
    if FractionalIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Fractional(TimeSeries, dim, fractional = True)],axis=1)
    if ActivationIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Activation(TimeSeries, dim, Sigmoid = True, Tanh = True, Regulation = True)],axis=1)
    
    return ElementaryMatrix

def Polynomial_functions(TimeSeries, dim, PolyOrder):
    Timelength = np.size(TimeSeries, 0)
    dim_multi_Nnodes = np.size(TimeSeries, 1)

    if PolyOrder >= 1:
        if dim == 1:
            lst = ['x1']
            Numfunc, Namefunc = elementary_functions_name(lst,1)
            PolyOne = np.zeros(shape=(Timelength,Numfunc))
            for j in range(0,Numfunc):
                PolyOne[:,j]  = TimeSeries[:,j]
            column_values = Namefunc
            PolyOne = pd.DataFrame(data = PolyOne, columns = column_values)

        if dim == 2:
            lst = ['x1','x2']
            Numfunc, Namefunc = elementary_functions_name(lst,1)
            PolyOne = np.zeros(shape=(Timelength,Numfunc))
            for j in range(0,Numfunc):
                PolyOne[:,j]  = TimeSeries[:,j]
            column_values = Namefunc
            PolyOne = pd.DataFrame(data = PolyOne, columns = column_values) 
        
        if dim == 3:
            lst = ['x1','x2','x3']
            Numfunc, Namefunc = elementary_functions_name(lst,1)
            PolyOne = np.zeros(shape=(Timelength,Numfunc))
            for j in range(0,Numfunc):
                PolyOne[:,j]  = TimeSeries[:,j]
            column_values = Namefunc
            PolyOne = pd.DataFrame(data = PolyOne, columns = column_values)

        if dim == 4:
            lst = ['x1','x2','x3','x4']
            Numfunc, Namefunc = elementary_functions_name(lst,1)
            PolyOne = np.zeros(shape=(Timelength,Numfunc))
            for j in range(0,Numfunc):
                PolyOne[:,j]  = TimeSeries[:,j]
            column_values = Namefunc
            PolyOne = pd.DataFrame(data = PolyOne, columns = column_values)


    if PolyOrder >= 2:
        if dim == 1:
            lst = ['x1']
            Numfunc, Namefunc = elementary_functions_name(lst,2)
            PolyTwo = np.zeros(shape=(Timelength,Numfunc))
            j = 0
            PolyTwo[:,j] = TimeSeries[:,j]**2
            column_values = Namefunc
            PolyTwo = pd.DataFrame(data = PolyTwo, columns = column_values)

        if dim == 2:
            lst = ['x1','x2']
            Numfunc, Namefunc = elementary_functions_name(lst,2)
            PolyTwo = np.zeros(shape=(Timelength,Numfunc))
            j = 0
            for ii in range(0,dim):
                for jj in range(ii,dim):
                    PolyTwo[:,j] = TimeSeries[:,ii]*TimeSeries[:,jj]
                    j = j+1
            column_values = Namefunc
            PolyTwo = pd.DataFrame(data = PolyTwo, columns = column_values)

        if dim == 3:
            lst = ['x1','x2','x3']
            Numfunc, Namefunc = elementary_functions_name(lst,2)
            PolyTwo = np.zeros(shape=(Timelength,Numfunc))
            j = 0
            for ii in range(0,dim):
                for jj in range(ii,dim):
                    PolyTwo[:,j] = TimeSeries[:,ii]*TimeSeries[:,jj]
                    j = j+1
            column_values = Namefunc
            PolyTwo = pd.DataFrame(data = PolyTwo, columns = column_values)

        if dim == 4:
            lst = ['x1','x2','x3','x4']
            Numfunc, Namefunc = elementary_functions_name(lst,2)
            PolyTwo = np.zeros(shape=(Timelength,Numfunc))
            j = 0
            for ii in range(0,dim):
                for jj in range(ii,dim):
                    PolyTwo[:,j] = TimeSeries[:,ii]*TimeSeries[:,jj]
                    j = j+1
            column_values = Namefunc
            PolyTwo = pd.DataFrame(data = PolyTwo, columns = column_values)

        
    if PolyOrder >= 3:
        if dim == 1:
            lst = ['x1']
            Numfunc, Namefunc = elementary_functions_name(lst,3)
            PolyThree = np.zeros(shape=(Timelength,Numfunc))
            j = 0
            for ii in range(0,dim):
                for jj in range(ii,dim):
                    for kk in range(jj,dim):
                        PolyThree[:,j] = TimeSeries[:,ii]*TimeSeries[:,jj]*TimeSeries[:,kk]
                        j = j+1
            column_values = Namefunc
            PolyThree = pd.DataFrame(data = PolyThree, columns = column_values)

        if dim == 2:
            lst = ['x1','x2']
            Numfunc, Namefunc = elementary_functions_name(lst,3)
            PolyThree = np.zeros(shape=(Timelength,Numfunc))
            j = 0
            for ii in range(0,dim):
                for jj in range(ii,dim):
                    for kk in range(jj,dim):
                        PolyThree[:,j] = TimeSeries[:,ii]*TimeSeries[:,jj]*TimeSeries[:,kk]
                        j = j+1
            column_values = Namefunc
            PolyThree = pd.DataFrame(data = PolyThree, columns = column_values)

        if dim == 3:
            lst = ['x1','x2','x3']
            Numfunc, Namefunc = elementary_functions_name(lst,3)
            PolyThree = np.zeros(shape=(Timelength,Numfunc))
            j = 0
            for ii in range(0,dim):
                for jj in range(ii,dim):
                    for kk in range(jj,dim):
                        PolyThree[:,j] = TimeSeries[:,ii]*TimeSeries[:,jj]*TimeSeries[:,kk]
                        j = j+1
            column_values = Namefunc
            PolyThree = pd.DataFrame(data = PolyThree, columns = column_values)

        if dim == 4:
            lst = ['x1','x2','x3','x4']
            Numfunc, Namefunc = elementary_functions_name(lst,3)
            PolyThree = np.zeros(shape=(Timelength,Numfunc))
            j = 0
            for ii in range(0,dim):
                for jj in range(ii,dim):
                    for kk in range(jj,dim):
                        PolyThree[:,j] = TimeSeries[:,ii]*TimeSeries[:,jj]*TimeSeries[:,kk]
                        j = j+1
            column_values = Namefunc
            PolyThree = pd.DataFrame(data = PolyThree, columns = column_values)

    if PolyOrder >= 4:
        if dim == 1:
            lst = ['x1']
            Numfunc, Namefunc = elementary_functions_name(lst,4)
            PolyFour = np.zeros(shape=(Timelength,Numfunc))
            j = 0
            for ii in range(0,dim):
                for jj in range(ii,dim):
                    for kk in range(jj,dim):
                        for ll in range(kk,dim):
                            PolyFour[:,j] = TimeSeries[:,ii]*TimeSeries[:,jj]*TimeSeries[:,kk]*TimeSeries[:,ll]
                            j = j+1
            column_values = Namefunc
            PolyFour = pd.DataFrame(data = PolyFour, columns = column_values)

        if dim == 2:
            lst = ['x1','x2']
            Numfunc, Namefunc = elementary_functions_name(lst,4)
            PolyFour = np.zeros(shape=(Timelength,Numfunc))
            j = 0
            for ii in range(0,dim):
                for jj in range(ii,dim):
                    for kk in range(jj,dim):
                        for ll in range(kk,dim):
                            PolyFour[:,j] = TimeSeries[:,ii]*TimeSeries[:,jj]*TimeSeries[:,kk]*TimeSeries[:,ll]
                            j = j+1
            column_values = Namefunc
            PolyFour = pd.DataFrame(data = PolyFour, columns = column_values)

        if dim == 3:
            lst = ['x1','x2','x3']
            Numfunc, Namefunc = elementary_functions_name(lst,4)
            PolyFour = np.zeros(shape=(Timelength,Numfunc))
            j = 0
            for ii in range(0,dim):
                for jj in range(ii,dim):
                    for kk in range(jj,dim):
                        for ll in range(kk,dim):
                            PolyFour[:,j] = TimeSeries[:,ii]*TimeSeries[:,jj]*TimeSeries[:,kk]*TimeSeries[:,ll]
                            j = j+1
            column_values = Namefunc
            PolyFour = pd.DataFrame(data = PolyFour, columns = column_values)

        if dim == 4:
            lst = ['x1','x2','x3','x4']
            Numfunc, Namefunc = elementary_functions_name(lst,4)
            PolyFour = np.zeros(shape=(Timelength,Numfunc))
            j = 0
            for ii in range(0,dim):
                for jj in range(ii,dim):
                    for kk in range(jj,dim):
                        for ll in range(kk,dim):
                            PolyFour[:,j] = TimeSeries[:,ii]*TimeSeries[:,jj]*TimeSeries[:,kk]*TimeSeries[:,ll]
                            j = j+1
            column_values = Namefunc
            PolyFour = pd.DataFrame(data = PolyFour, columns = column_values)
    if PolyOrder == 1:
        return PolyOne
    if PolyOrder == 2:
        return pd.concat([PolyOne, PolyTwo], axis=1)
    if PolyOrder == 3:
        return pd.concat([PolyOne, PolyTwo, PolyThree], axis=1)
    if PolyOrder == 4:
        return pd.concat([PolyOne, PolyTwo, PolyThree, PolyFour], axis=1)

def Trigonometric(TimeSeries, dim, Sin = True, Cos = True, Tan = True):
    Timelength = np.size(TimeSeries, 0)
    dim_multi_Nnodes = np.size(TimeSeries, 1)

    if Sin == True:
        sine = np.zeros(shape=(Timelength,dim))
        for j in range(0,dim):
            sine[:,j] = np.sin(TimeSeries[:,j])
        if dim == 1:
            column_values = ['sinx1']
        if dim == 2:
            column_values = ['sinx1','sinx2']
        if dim == 3:
            column_values = ['sinx1','sinx2','sinx3']
        if dim == 4:
            column_values = ['sinx1','sinx2','sinx3','sinx4']
        sine = pd.DataFrame(data = sine, columns = column_values)

    if Cos == True:
        cosine = np.zeros(shape=(Timelength,dim))
        for j in range(0,dim):
            cosine[:,j] = np.cos(TimeSeries[:,j])
        if dim == 1:
            column_values = ['cosx1']
        if dim == 2:
            column_values = ['cosx1','cosx2']
        if dim == 3:
            column_values = ['cosx1','cosx2','cosx3']
        if dim == 4:
            column_values = ['cosx1','cosx2','cosx3','cosx4']
        cosine = pd.DataFrame(data = cosine, columns = column_values)

    if Tan == True:
        tangent = np.zeros(shape=(Timelength,dim))
        for j in range(0,dim):
            tangent[:,j] = np.tan(TimeSeries[:,j])
        if dim == 1:
            column_values = ['tanx1']
        if dim == 2:
            column_values = ['tanx1','tanx2']
        if dim == 3:
            column_values = ['tanx1','tanx2','tanx3']
        if dim == 4:
            column_values = ['tanx1','tanx2','tanx3','tanx4']
        tangent= pd.DataFrame(data = tangent, columns = column_values)
    return pd.concat([sine, cosine, tangent], axis=1)

def Exponential(TimeSeries, dim, expomential = True):
    Timelength = np.size(TimeSeries, 0)
    dim_multi_Nnodes = np.size(TimeSeries, 1)
    if expomential == True:
        Exp = np.zeros(shape=(Timelength,dim))
        for j in range(0,dim):
            Exp[:,j] = np.exp(TimeSeries[:,j])
        if dim == 1:
            column_values = ['expx1']
        if dim == 2:
            column_values = ['expx1','expx2']
        if dim == 3:
            column_values = ['expx1','expx2','expx3']
        if dim == 4:
            column_values = ['expx1','expx2','expx3','expx4']
        Exp = pd.DataFrame(data = Exp, columns = column_values)
    return Exp

def Fractional(TimeSeries, dim, fractional = True):
    Timelength = np.size(TimeSeries, 0)
    dim_multi_Nnodes = np.size(TimeSeries, 1)
    if fractional == True:
        frac = np.zeros(shape=(Timelength,dim))
        for j in range(0,dim):
            frac[:,j] = 1/TimeSeries[:,j]
        if dim == 1:
            column_values = ['fracx1']
        if dim == 2:
            column_values = ['fracx1','fracx2']
        if dim == 3:
            column_values = ['fracx1','fracx2','fracx3']
        if dim == 4:
            column_values = ['fracx1','fracx2','fracx3','fracx4']
        frac = pd.DataFrame(data = frac, columns = column_values)
    return frac

def Activation(TimeSeries, dim, Sigmoid = True, Tanh = True, Regulation = True):
    Timelength = np.size(TimeSeries, 0)
    dim_multi_Nnodes = np.size(TimeSeries, 1)
    if Sigmoid == True:
        #alpha = np.linspace(1,10,10)
        alpha = [1, 5, 10]
        #beta = np.linspace(0,10,11)
        beta = [0,1, 5, 10]
        Numfunc = len(alpha)*len(beta)
        sigmoid = np.zeros(shape=(Timelength,dim*Numfunc))
        kk = 0
        for j in range(0,dim):
            for ii in range(0,len(alpha)):
                for jj in range(0,len(beta)):
                    sigmoid[:,kk] = 1./(1.+np.exp(-alpha[ii]*(TimeSeries[:,j]-beta[jj])))
                    kk = kk+1
        kk = 0
        Sigmoid = pd.DataFrame()
        for j in range(0,dim):
            for ii in range(0,len(alpha)):
                for jj in range(0,len(beta)):
                    tmp = ["sig_x"+str(j+1)+"_"+str(alpha[ii])+str(beta[jj])]
                    tmp_2 = pd.DataFrame(data = sigmoid[:,kk], columns = tmp)
                    kk = kk+1
                    Sigmoid = pd.concat([Sigmoid,tmp_2], axis=1)
    
    if Tanh == True:
        tanh = np.zeros(shape=(Timelength,dim))
        for j in range(0,dim):
            tanh[:,j] = (np.exp(TimeSeries[:,j])-np.exp(-TimeSeries[:,j]))/(np.exp(TimeSeries[:,j])+np.exp(-TimeSeries[:,j]))
        if dim == 1:
            column_values = ['tanhx1']
        if dim == 2:
            column_values = ['tanhx1','tanhx2']
        if dim == 3:
            column_values = ['tanhx1','tanhx2','tanhx3']
        if dim == 4:
            column_values = ['tanhx1','tanhx2','tanhx3','tanhx4']
        tanh = pd.DataFrame(data = tanh, columns = column_values)

    if Regulation == True:
        #gamma = np.linspace(0,10,11)
        gamma = [1,2,5,10]
        Numfunc = len(gamma)
        regulation = np.zeros(shape=(Timelength,dim*Numfunc))
        kk = 0
        for j in range(0,dim):
            for ii in range(0,len(gamma)):
                    regulation[:,kk] = (TimeSeries[:,j]**gamma[ii])/(TimeSeries[:,j]**gamma[ii]+1)
                    kk = kk+1

        kk = 0
        Regulation = pd.DataFrame()
        for j in range(0,dim):
            for ii in range(0,len(gamma)):
                tmp = ["regx"+str(j+1)+"_"+str(gamma[ii])]
                tmp_2 = pd.DataFrame(data = regulation[:,kk], columns = tmp)
                kk = kk+1
                Regulation = pd.concat([Regulation,tmp_2], axis=1)
    return pd.concat([Sigmoid, tanh, Regulation], axis=1)