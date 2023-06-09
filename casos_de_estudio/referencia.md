# Aqui se explica cada una de las lineas de codigo de 
Realizar esto luego de delcarar el path en maltab de [mmclab](https://github.com/fangq/mmc/tree/master/mmclab) anteriormente ademas de [iso2mesh](https://github.com/fangq/iso2mesh) del mismo repositorio
### [demo_mmclab_basic.m](https://github.com/fangq/mmc/blob/master/mmclab/example/demo_immc_basic.m)

## clear cfg

limpia la variable cfg de cualquier valor anterior.

## cfg.nphoton=1e6;

establece el número de fotones a 1 millón.

## [cfg.node, face, cfg.elem]=meshabox([0 0 0],[60 60 30],6);

Crea una malla de cajas (mesh) 3D usando la función meshabox que toma dos puntos opuestos (esquina inferior izquierda y esquina superior derecha) y el tamaño de la caja. El número "6" se refiere al tamaño del elemento en la malla.

## cfg.elemprop=ones(size(cfg.elem,1),1);

Define las propiedades de los elementos de la malla como "1" para todos los elementos.

## cfg.srcpos=[30 30 0];

Establece la posición de la fuente de luz en (30,30,0) en la malla.

## cfg.srcdir=[0 0 1];

Establece la dirección de la fuente de luz en la dirección z (en este caso, la fuente de luz está en el plano xy y apunta hacia el eje z).

## cfg.prop=[0 0 1 1;0.005 1 0 1.37];

Establece las propiedades ópticas del medio. El primer elemento corresponde a la absorción y el segundo elemento corresponde al coeficiente de dispersión (o esparcimiento). En este caso, el medio es completamente transparente (absorción = 0) y tiene un coeficiente de dispersión de 0.005 mm^-1 con un índice de refracción de 1.37.

## cfg.tstart=0;

Establece el tiempo inicial de la simulación en cero.

## cfg.tend=5e-9;

Establece el tiempo final de la simulación en 5 nanosegundos.

## cfg.tstep=5e-9;

Establece el tamaño del paso de tiempo en 5 nanosegundos.

## cfg.debuglevel='TP';

Establece el nivel de depuración de la simulación en 'TP' (Time and Photon).

## cfg.issaveref=1;

Establece si se deben guardar los datos de reflectancia difusa en la superficie (surface diffuse reflectance).

## cfg.method='elem';

Establece el método de simulación de Monte Carlo como "elem" (para elementos).

## flux=mmclab(cfg);

Ejecuta la simulación y guarda los datos de fluencia (fluence) en la variable flux.

## subplot(121);

Divide la ventana de visualización de MATLAB en 2 columnas y 1 fila y selecciona el primer panel.

## plotmesh([cfg.node(:,1:3),log10(abs(flux.data(1:size(cfg.node,1))))],cfg.elem,'y=30','facecolor','interp','linestyle','none')

Grafica la sección transversal de la fluencia en el plano y=30 usando la función plotmesh. Los primeros tres elementos de cfg.node corresponden a las coordenadas x, y, z de cada