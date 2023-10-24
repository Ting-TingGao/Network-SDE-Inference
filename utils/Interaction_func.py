import pandas as pd
import numpy as np

def ElementaryFunctions_Matrix(xi, xj, coupledPolyOrder = 1, CoupledPolynomialIndex = True, \
        CoupledTrigonometricIndex = True, CoupledExponentialIndex = True, CoupledFractionalIndex = True, \
            CoupledActivationIndex = True):
    
    ElementaryMatrix = pd.DataFrame()
    if CoupledPolynomialIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Coupled_Polynomial_functions(xi, xj, coupledPolyOrder)],axis=1)
    if CoupledTrigonometricIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Coupled_Trigonometric_functions(xi, xj, Sine = True, Cos = False, Tan = False)],axis=1)
    if CoupledExponentialIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Coupled_Exponential_functions(xi, xj, Exponential = True)],axis=1)
    if CoupledFractionalIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Coupled_Fractional_functions(xi, xj, Fractional = True)],axis=1)
    if CoupledActivationIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Coupled_Activation_functions(xi, xj, Sigmoid = True, Tanh = True, Hill = True)],axis=1)
   
    return ElementaryMatrix


#Libraries construction
def Coupled_Polynomial_functions(xi,xj,PolyOrder):
    if PolyOrder>=1:
        column_values = ['xj','xixj','xjMinusxi']
        CoupledPolyOne = np.zeros(shape=(np.size(xi,0),len(column_values)))
        for j in range(len(column_values)):
            if j == 0:
                tmp1 = xj
            if j == 1:
                tmp2 = xi*xj
            if j == 2:
                tmp3 = xj-xi
        CoupledPolyOne = np.stack((tmp1,tmp2,tmp3),axis=1)
        CoupledPolyOne = pd.DataFrame(data = CoupledPolyOne, columns = column_values)
    if PolyOrder>=2:
        column_values = ['xjpow2','xixjpow2','xjMinusxipow2']
        CoupledPolyTwo = np.zeros(shape=(np.size(xi,0),len(column_values)))
        for j in range(len(column_values)):
            if j == 0:
                tmp1 = xj**2
            if j == 1:
                tmp2 = (xi*xj)**2
            if j == 2:
                tmp3 = (xj-xi)**2
        CoupledPolyTwo = np.stack((tmp1,tmp2,tmp3),axis=1)
        CoupledPolyTwo = pd.DataFrame(data = CoupledPolyTwo, columns = column_values)
    if PolyOrder == 1:
        return CoupledPolyOne
    if PolyOrder == 2:
        return pd.concat([CoupledPolyOne, CoupledPolyTwo], axis=1)

def Coupled_Trigonometric_functions(xi, xj, Sine = True, Cos = False, Tan = False):
    if Sine == True:
        column_values = ['sinxj','sinxixj','sinxjMinusxi','xisinxj']
        CoupledSine = np.zeros(shape=(np.size(xi,0),len(column_values)))
        for j in range(len(column_values)):
            if j == 0:
                tmp1 = np.sin(xj)
            if j == 1:
                tmp2 = np.sin(xi*xj)
            if j == 2:
                tmp3 = np.sin(xj-xi)
            if j == 3:
                tmp4 = xi*np.sin(xj)
        CoupledSine = np.stack((tmp1,tmp2,tmp3,tmp4),axis=1)
        CoupledSine = pd.DataFrame(data = CoupledSine, columns = column_values)
    if Cos == True:
        column_values = ['cosxj','cosxixj','cosxjMinusxi','xicosxj']
        CoupledCosine = np.zeros(shape=(np.size(xi,0),len(column_values)))
        for j in range(len(column_values)):
            if j == 0:
                tmp1 = np.cos(xj)
            if j == 1:
                tmp2 = np.cos(xi*xj)
            if j == 2:
                tmp3 = np.cos(xj-xi)
            if j == 3:
                tmp4 = xi*np.cos(xj)
        CoupledCosine = np.stack((tmp1,tmp2,tmp3,tmp4),axis=1)
        CoupledCosine = pd.DataFrame(data = CoupledCosine, columns = column_values)
    if Tan == True:
        column_values = ['tanxj','tanxixj','tanxjMinusxi','xitanxj']
        CoupledTan = np.zeros(shape=(np.size(xi,0),len(column_values)))
        for j in range(len(column_values)):
            if j == 0:
                tmp1 = np.tan(xj)
            if j == 1:
                tmp2 = np.tan(xi*xj)
            if j == 2:
                tmp3 = np.tan(xj-xi)
            if j == 3:
                tmp4 = xi*np.tan(xj)
        CoupledTan = np.stack((tmp1,tmp2,tmp3,tmp4),axis=1)
        CoupledTan = pd.DataFrame(data = CoupledTan, columns = column_values)
    CoupledTrignometric = pd.DataFrame()
    if Sine == True:
        CoupledTrignometric = pd.concat([CoupledTrignometric,CoupledSine],axis=1)
    if Cos == True:
        CoupledTrignometric = pd.concat([CoupledTrignometric, CoupledCosine], axis=1)
    if Tan == True:
        CoupledTrignometric = pd.concat([CoupledTrignometric, CoupledTan], axis=1)
    return CoupledTrignometric

def Coupled_Exponential_functions(xi, xj, Exponential = True):
    if Exponential == True:
        column_values = ['expxj','expxixj','expxjMinusxi','xiexpxj']
        CoupledExp = np.zeros(shape=(np.size(xi,0),len(column_values)))
        for j in range(len(column_values)):
            if j == 0:
                tmp1 = np.exp(xj)
            if j == 1:
                tmp2 = np.exp(xi*xj)
            if j == 2:
                tmp3 = np.exp(xj-xi)
            if j == 3:
                tmp4 = xi*np.exp(xj)
        CoupledExp = np.stack((tmp1,tmp2,tmp3,tmp4),axis=1)    
        CoupledExp = pd.DataFrame(data = CoupledExp, columns = column_values)
    return CoupledExp

def Coupled_Fractional_functions(xi, xj, Fractional = True):
    if Fractional == True:
        column_values = ['fracxj','fracxixj','fracxjMinusxi','xifracxj']
        CoupledFraction = np.zeros(shape=(np.size(xi,0),len(column_values)))
        for j in range(len(column_values)):
            if j == 0:
                tmp1 = 1/xj
            if j == 1:
                tmp2 = 1/(xi*xj)
            if j == 2:
                tmp3 = 1/((xj-xi)+1e-5)
            if j == 3:
                tmp4 = xi/(xj+1e-5)
        CoupledFraction = np.stack((tmp1,tmp2,tmp3,tmp4),axis=1) 
        CoupledFraction = pd.DataFrame(data = CoupledFraction, columns = column_values)
    return CoupledFraction

def sigmoidfun(x,alpha,beta):
    sigmoidOutput = 1/(1+np.exp(-alpha*(x-beta)))
    return sigmoidOutput

def tangentH(x):
    Tanh = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return Tanh

def Hill_func(x,gamma):
    Regulation_result = (x**gamma)/(x**gamma+1)
    return Regulation_result

def Coupled_Activation_functions(xi, xj, Sigmoid = True, Tanh = True, Hill = True):
    if Sigmoid == True:
        column_values = ['sigmoidxj','sigmoidxixj','sigmoidXjMinusXi','xisigmoidxj',
                        'sigmoidxj101','sigmoidxixj101','sigmoidXjMinusXi101','xisigmoidxj101']
        CoupledSigmoid = np.zeros(shape=(np.size(xi,0),len(column_values)))
        for j in range(len(column_values)):
            if j == 0:
                tmp1 = sigmoidfun(xj,1,0)
            if j == 1:
                tmp2 = sigmoidfun(xi*xj,1,0)
            if j == 2:
                tmp3 = sigmoidfun(xj-xi,1,0)
            if j == 3:
                tmp4 = sigmoidfun(xj,1,0)*xi
            if j == 4:
                tmp5 = sigmoidfun(xj,10,1)
            if j == 5:
                tmp6 = sigmoidfun(xi*xj,10,1)
            if j == 6:
                tmp7 = sigmoidfun(xj-xi,10,1)
            if j == 7:
                tmp8 = sigmoidfun(xj,10,1)*xi
        CoupledSigmoid = np.stack((tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8),axis=1)
        CoupledSigmoid = pd.DataFrame(data = CoupledSigmoid, columns = column_values)
    if Tanh == True:
        column_values = ['tanhxj','tanhxixj','tanhxjMinusxi','xitanhxj']
        CoupledTanh = np.zeros(shape=(np.size(xi,0),len(column_values)))
        for j in range(len(column_values)):
            if j == 0:
                tmp1 = tangentH(xj)
            if j == 1:
                tmp2 = tangentH(xi*xj)
            if j == 2:
                tmp3 = tangentH(xj-xi)
            if j == 3:
                tmp4 = tangentH(xj)*xi
        Coupledtanh = np.stack((tmp1,tmp2,tmp3,tmp4),axis=1)
        Coupledtanh = pd.DataFrame(data = Coupledtanh, columns = column_values)
    if Hill == True:
        column_values = ['hillxj','hillxixj','hillxjMinusxi','xihillxj','hillxj2',
                        'hillxixj2','hillxjMinusxi2','hillxj5','hillxixj5','hillxjMinusxi5']
        CoupledHill = np.zeros(shape=(np.size(xi,0),len(column_values)))
        for j in range(len(column_values)):
            if j == 0:
                tmp1 = Hill_func(xj,1)
            if j == 1:
                tmp2 = Hill_func(xi*xj,1)
            if j == 2:
                tmp3 = Hill_func(xj-xi,1)
            if j == 3:
                tmp4 = Hill_func(xj,1)*xi
            if j == 4:
                tmp5 = Hill_func(xj,2)
            if j == 5:
                tmp6 = Hill_func(xi*xj,2)
            if j == 6:
                tmp7 = Hill_func(xj-xi,2)
            if j == 7:
                tmp8 = Hill_func(xj,5)
            if j == 8:
                tmp9 = Hill_func(xi*xj,5)
            if j == 9:
                tmp10 = Hill_func(xj-xi,5)
        CoupledHill = np.stack((tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9,tmp10),axis=1)
        CoupledHill = pd.DataFrame(data = CoupledHill, columns = column_values)   
    return pd.concat([CoupledSigmoid,Coupledtanh,CoupledHill], axis = 1) 