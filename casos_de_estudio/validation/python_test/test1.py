import math
def mmclab(cfg, **options):
    # Obtener los parámetros de configuración
    source_intensity = cfg['source_intensity']
    absorption_coefficient = cfg['absorption_coefficient']
    distance = cfg['distance']

    # Realizar el cálculo de intensidad (de forma especifica irradiancia (w/m^2)
    #source_intensity esta en w/m^2
    intensity = source_intensity * math.exp(-absorption_coefficient * distance) #1/m y m


    # Devolver los resultados de la simulación
    return intensity
cfg = {
    'source_intensity': 10,
    'absorption_coefficient': 0.5,
    'distance': 2.5
}
resultado = mmclab(cfg)
print("la irradiancia es: ",resultado)

