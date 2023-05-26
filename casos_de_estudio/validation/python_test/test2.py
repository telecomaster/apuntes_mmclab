# it is my first try, that is why i made it as spaguetti,update very soon :)

import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import griddata


# Configuración de parámetros
cfg = {
    'nphoton': 3e6,
    'seed': 1648335518,
    'node': None,
    'elem': None,
    'elemprop': None,
    'srcpos': [30.1, 30.2, 0],
    'srcdir': [0, 0, 1],
    'tstart': 0,
    'tend': 5e-9,
    'tstep': 1e-10,
    'prop': [[0, 0, 1, 1], [0.005, 1.0, 0.01, 1.0]],
    'debuglevel': 'TP',
    'isreflect': 0,
    'method': 'elem'
}

def qmeshcut(elements, nodes, cutvalue, plane):
    tri = Delaunay(nodes)
    simplex_indices = tri.find_simplex(plane)
    
    if np.any(simplex_indices < 0):
        raise ValueError("The plane is outside the mesh.")
    
    simplex_vertices = tri.simplices[simplex_indices]
    
    cutpos = []
    for vertex_indices in simplex_vertices:
        element_nodes = nodes[elements[vertex_indices]]
        weights = tri.transform[simplex_indices, :3].dot((plane - tri.transform[simplex_indices, 3])[:, None])
        cutpos.append(np.average(element_nodes, weights=weights, axis=1))
    
    cutpos = np.concatenate(cutpos)
    cutvalue = cutvalue[simplex_vertices.flatten()]
    
    return cutpos, cutvalue


def genT5mesh(*args):

    n = len(args)
    if n != 3:
        raise ValueError("solo para 3D")
    
    for i in range(n):
        v = args[i]
        if len(v) % 2 == 0:
            args[i] = np.linspace(v[0], v[-1], len(v) + 1)
    
    cube8 = np.array([
        [1, 4, 5, 13], [1, 2, 5, 11], [1, 10, 11, 13], [11, 13, 14, 5], [11, 13, 1, 5],
        [2, 3, 5, 11], [3, 5, 6, 15], [15, 11, 12, 3], [15, 11, 14, 5], [11, 15, 3, 5],
        [4, 5, 7, 13], [5, 7, 8, 17], [16, 17, 13, 7], [13, 17, 14, 5], [5, 7, 17, 13],
        [5, 6, 9, 15], [5, 8, 9, 17], [17, 18, 15, 9], [17, 15, 14, 5], [17, 15, 5, 9],
        [10, 13, 11, 19], [13, 11, 14, 23], [22, 19, 23, 13], [19, 23, 20, 11], [13, 11, 19, 23],
        [11, 12, 15, 21], [11, 15, 14, 23], [23, 21, 20, 11], [23, 24, 21, 15], [23, 21, 11, 15],
        [16, 13, 17, 25], [13, 17, 14, 23], [25, 26, 23, 17], [25, 22, 23, 13], [13, 17, 25, 23],
        [17, 18, 15, 27], [17, 15, 14, 23], [26, 27, 23, 17], [27, 23, 24, 15], [23, 27, 17, 15]
    ]).T - 1
    
    nodecount = [len(arg) for arg in args]
    if any(np.array(nodecount) < 2):
        raise ValueError("Each dimension must be of size 2 or more.")
    
    vertices = lattice(*args)
    
#nota 2 si mi memoria no falla es la velocidad de la luz en m/s
c0=299792458000;

# Generar malla
x = np.arange(0, 62, 2)
y = np.arange(0, 62, 2)
z = np.arange(0, 62, 2)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
cfg['node'] = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

# Generar propiedades de los elementos
cfg['elem'] = np.arange(1, len(cfg['node'])+1)
cfg['elemprop'] = np.ones(len(cfg['elem']))

# Función para realizar la simulación
def mmclab(cfg):
    # Aquí iría la implementación del algoritmo MMCM
    # Obtener los parámetros de configuración
    source_intensity = cfg['source_intensity']
    absorption_coefficient = cfg['absorption_coefficient']
    distance = cfg['distance']

    # Realizar el cálculo de intensidad (de forma especifica irradiancia (w/m^2)
    #source_intensity esta en w/m^2
    intensity = source_intensity * math.exp(-absorption_coefficient * distance) #1/m y m
    result = {'data': intensity}  # 'intensity' es el resultado de la simulación

    # Devolver los resultados de la simulación
    return result

#declaro mi diccionario
cfg = {
    'source_intensity': 10,
    'absorption_coefficient': 0.5,
    'distance': 2.5,
    'tstart': 0.0,  # Agregar la clave 'tstart' con su valor correspondiente
    'tstep' : 0.0000000001, 
    'tend' : 0.000000005,
    #'prop':[0 0 1 1;0.005 1.0 0.01 1.0],
    #'debuglevel'='TP'
    'isreflect':0,
    #cfg.method='elem'


}

#esta funcion es inherente de matlab pero la tengo que declarar en python


def getdistance(srcpos, detpos):
    diff = detpos - srcpos
    distance = np.sqrt(np.sum(diff**2, axis=1))
    return distance

#esta es una funcion del paquete mcx de mcclab

def tddiffusion(mua, musp, v, Reff, srcpos, detpos, t):
    D = 1 / (3 * (mua + musp))
    zb = (1 + Reff) / (1 - Reff) * 2 * D

    z0 = 1 / (musp + mua)
    #aqui una variacion con matlab, structura de datos de srcpos es unidimensional en Python, mientras que en Matlab se trata como una matriz 2D
    #Esta línea primero utiliza np.reshape para cambiar la forma de srcpos a (-1, 3), lo que indica que queremos mantener la dimensión de las filas y tener 3 columnas. Luego, 
    # se utiliza np.column_stack y el cálculo de r para obtener los resultados deseados.
    srcpos = np.reshape(srcpos, (-1, 3))
    r = getdistance(np.column_stack((srcpos[:, :2], srcpos[:, 2] + z0)), detpos)

    #con esto se tira r = getdistance(np.column_stack((srcpos[:, :2], srcpos[:, 2:] + z0)), detpos)

    r2 = getdistance(np.column_stack((srcpos[:, 0:2], srcpos[:, 2] - z0 - 2 * zb)), detpos)

    s = 4 * D * v * t

    # unit of phi:  1/(mm^2*s)
    Phi = v / ((s * np.pi) ** (3 / 2)) * np.exp(-mua * v * t) * (np.exp(-(r ** 2) / s) - np.exp(-(r2 ** 2) / s))
    cfg['node'] = np.array([[30, 14, 10]])  # Agrega la clave 'node' con su valor correspondiente, es la posicion del detector


    return Phi



    #pass

# Ejecutar simulación
cube = mmclab(cfg)
cube = cube['data']  # Acceder al atributo 'data' del diccionario 'cube'

#cube = cube.data

# Calcular TPSF en un punto específico
twin = np.arange(cfg['tstart'] + cfg['tstep']/2, cfg['tend'], cfg['tstep'])
gates = len(twin)

srcpos = np.array([30.1, 30.2, 0])
detpos = np.array([30, 14, 10])

tpsf_diffusion = np.zeros(gates)
tpsf_mmcm = np.zeros(gates)

#no se si esta bien la aplicacion de tddi pero aqui voy a declarar


# ... Código anterior ...

# Llamada a la función tddiffusion
Phi = tddiffusion(0.005, 1, c0, 0, srcpos, detpos, twin)

plt.figure()

srcpos = np.array([30.1, 30.2, 0])
detpos = np.array([30, 14, 10])


plt.semilogy(np.arange(1, gates + 1) / 10, cube[idx, :], '+')

#plt.semilogy(np.arange(1, gates + 1) / 10, Phi, 'r')  # Utiliza el resultado de la función tddiffusion

idx = np.where(np.isin(cfg['node'], detpos, assume_unique=True))[0]
plt.semilogy(np.arange(1, gates + 1) / 10, cube[idx, :], '+')


plt.xlabel('t (ns)')
plt.ylabel('Fluence TPSF (1/mm^2)')
plt.yscale('log')
plt.legend(['Diffusion', 'MMCM'])
plt.legend(boxoff=True)
plt.box(True)
# ... Código posterior ...



for i in range(gates):
    tpsf_diffusion[i] = tddiffusion(0.005, 1, c0, 0, srcpos, detpos, twin[i])
    idx = np.where((cfg['node'] == detpos).all(axis=1))[0][0]
    tpsf_mmcm[i] = cube[idx]

# Graficar TPSF en función del tiempo
plt.figure()
plt.semilogy(np.arange(1, gates+1)/10, tpsf_diffusion, 'r')
plt.semilogy(np.arange(1, gates+1)/10, tpsf_mmcm, '+')
plt.xlabel('t (ns)')
plt.ylabel('Fluence TPSF (1/mm^2)')
plt.yscale('log')
plt.legend(['Diffusion', 'MMCM'])
plt.grid(True)
plt.show()

# Generar gráfico de contorno
cutpos = qmeshcut(cfg['elem'][:, :4], cfg['node'], cwcb, [[0, 30.2, 0], [0, 30.2, 1], [1, 30.2, 0]])
xi, yi = np.meshgrid(x, y)
vi = griddata(cutpos[:, 0], cutpos[:, 2], cutvalue, xi, yi)

plt.figure()
clines = np.arange(-1.5, -8.5, -0.5)
plt.contour(xi, yi, np.log10(np.maximum(np.squeeze(phicw), 1e-8)), levels=clines, colors='k')
plt.contour(xi, yi, np.log10(np.abs(vi * cfg['tstep'])), levels=clines, colors='r', linestyles=':')
plt.axis('equal')
plt.xlim(1, 60)
plt.xlabel('x (mm)')
plt.ylabel('z (mm)')
plt.legend(['Diffusion', 'MMC'])
plt.grid(True)
plt.show()
