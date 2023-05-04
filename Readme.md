# Nociones a tomar en cuenta antes de trabajar con este toolkit

*Tener una tarjeta de video dedicada NVIDIA RTX (se hicieron pruebas con RTX 3060 y GTX 1060, la última 
aconseja que se deba actualizar por  tiempos de ejecución)

*NVIDIA CUDA toolkit instalado, no es obligatorio pero a futuro puede que se necesite.

*Se puede trabajar en entorno de LINUX, ejemplo Ubuntu, sin embargo estas pruebas se realizaron en Windows. (W11)

# Bienvenida 8-)

¡Hola! Bienvenidos al archivo readme.md para el toolkit de matlab MMCLAB.

Este archivo tiene como objetivo proporcionar información sobre cómo utilizar el MMCLAB de manera efectiva. El MMCLAB es una herramienta muy poderosa que permite la simulación de sistemas de comunicación mediante el uso de Matlab. Con esta herramienta, se puede diseñar, simular y analizar sistemas de comunicaciones digitales y analógicos.

A continuación, se presentan los pasos básicos para comenzar a utilizar el MMCLAB:

## Instalación:
Antes de poder utilizar el MMCLAB, debe estar instalado en su sistema. Para hacerlo, simplemente descargue la última versión del MMCLAB y siga las instrucciones de instalación que se encuentran en el archivo "README.txt" incluido en el paquete.

## Inicio del MMCLAB:
Una vez que haya instalado el MMCLAB, abra Matlab y escriba:
 
    mmc_init 
  // Some comments
    line 1 of code
    line 2 of code
    line 3 of code
 
 en la línea de comandos. Esto iniciará el MMCLAB y estará listo para su uso.

## Carga de muestras y configuración del sistema:
Para cargar muestras, simplemente use la función "mmcread" que carga los datos desde un archivo. Luego, use la función "mmcsetup" para configurar el sistema de comunicación con los parámetros necesarios.

## Ejecución de simulación:
Para ejecutar una simulación, use la función "mmcexec". Esto ejecutará la simulación y generará los resultados correspondientes.

## Visualización de resultados:
Los resultados generados se pueden visualizar mediante el uso de las funciones de Matlab para gráficos y visualización de datos. Para obtener ayuda adicional, consulte la documentación del MMCLAB y las funciones de ayuda en Matlab.

### Ahora ya estas listo para comenzar a utilizar  