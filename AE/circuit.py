
from qiskit.circuit import *

    
def makeSine(circuit, q, n, b):
    circuit.ry(b/(2**n),q[0])
    
    for i in range(n):
        circuit.cry(b/(2**(n-i-1)), q[i+1], q[0])
    
def makeSineInv(circuit, q, n, b): 
    
    for i in range(n-1, -1, -1):
        circuit.cry(- b/(2**(n-i-1)), q[i+1], q[0])
       
    circuit.ry(- b/(2**n),q[0])

    
def addH(circuit, q, n):
    for i in range(n):
        circuit.h(q[i+1])

    
def addA(circuit, q, n, b, barrier):
    addH(circuit, q, n)
    if barrier:
        circuit.barrier(q)
    makeSine(circuit, q, n, b)

def addAinv(circuit, q, n, b, barrier):
    makeSineInv(circuit, q, n, b)
    if barrier:
        circuit.barrier(q)
    addH(circuit, q, n)

def addSx(circuit, q):
    circuit.z(q[0])
    
def addS0(circuit, q, n):
   
    for i in range(n+1):
        circuit.x(q[i])
    circuit.h(q[0])
    circuit.mct([q[k] for k in range(n, 0, -1)],q[0], None, mode='noancilla')
    circuit.h(q[0])
    for i in range(n+1):
        circuit.x(q[i])
    
    
def addQ(circuit, q, n, b, barrier):
    if barrier:
        circuit.barrier(q)
        
    addSx(circuit, q)
    
    if barrier:
        circuit.barrier(q)
        
    addAinv(circuit, q, n, b, barrier)
    
    if barrier:
        circuit.barrier(q)

    addS0(circuit, q, n)
    
    if barrier:
        circuit.barrier(q)

    addA(circuit, q, n, b, barrier)

    


