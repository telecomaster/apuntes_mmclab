# notas sobre el algoritmo
El siguiente código de MATLAB es un ejemplo de cómo validar el algoritmo MMCM (Monte Carlo Modified) utilizando un dominio cúbico homogéneo. Este código genera los resultados mostrados en la Figura 2 del artículo del MMCM.

Primero se establecen las propiedades del dominio cúbico, que tiene una dimensión de 60x60x60 mm y propiedades ópticas de 
	mua=0.001 
	mus=1 
	n=1.0 
	g=0.01 

Luego, se genera una malla tetraédrica utilizando la función "genT5mesh" y se establecen las propiedades de los elementos.

Se configuran los parámetros de la simulación, como el número de fotones (cfg.nphoton), la posición y dirección de la fuente de luz (cfg.srcpos y cfg.srcdir), y el tiempo de inicio, final y paso (cfg.tstart, cfg.tend y cfg.tstep). También se establecen otros parámetros como el nivel de depuración (cfg.debuglevel) y el método utilizado (cfg.method).

Se utiliza la función "mmclab" para ejecutar la simulación utilizando la configuración anterior. La variable "cube" contiene los resultados de la simulación.

Luego, se calcula el tiempo de propagación de la luz (TPSF) en un punto específico utilizando la función "tddiffusion" y se traza la curva TPSF en función del tiempo. También se traza la curva TPSF obtenida por la simulación MMCM. Se genera un gráfico con el resultado.

Finalmente, se genera un mapa de contorno a lo largo de la línea y=30.2. Se traza el mapa de contorno obtenido por la simulación de difusión y el mapa de contorno obtenido por la simulación MMCM. Se genera un gráfico con el resultado.
