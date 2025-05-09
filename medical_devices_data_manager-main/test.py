import numpy as np
import matplotlib.pyplot as plt

from service.device1_service import *

if __name__ == '__main__':
    pulse_wave_data = [-57, -8, 43, 84, 102, 93, 61, 17, -28, -64, -85, -92, -89, -80, -70, -61, -56, -56, -62, -72,
                       -84, -94, -99, -100, -98, -95, -91, -88, -86, -86, -85, -84, -85, -90, -99, -107, -105, -87, -52,
                       -3, 45, 82, 96, 84, 51, 8, -34, -66, -84, -88, -83, -75, -67, -61, -59, -62, -69, -79, -88, -95,
                       -99, -98, -96, -92, -89, -89, -88, -88, -88, -89, -92, -99, -108, -111, -100, -71, -26, 25, 70,
                       96, 95, 70, 28, -18, -58, -84, -94, -92, -84, -72, -63, -57, -58, -64, -75, -86, -95, -99, -99,
                       -95, -92, -89, -88, -89, -90, -90, -90, -89, -92, -98, -106, -109, -98, -69, -24, 25, 68, 92, 89,
                       62, 18, -28, -68, -92, -101, -97, -75, -66, -61, -62, -70, -81, -92, -100, -104, -103, -99, -94,
                       -90, -87, -87, -87, -88, -89, -91, -97, -105, -109, -101, -74, -32, 19, 68, 98, 102, 80, 38, -10,
                       -53, -83, -97, -96, -87, -75, -65, -57, -55, -60, -70, -83, -94, -102, -105, -104, -101, -97,
                       -93, -91, -90, -89, -87, -88, -94, -103, -110, -106, -83, -41, 12, 96, 105, 87, 48, 1, -41, -73,
                       -88, -90, -84, -76, -67, -61, -59, -63, -71, -81, -91, -98, -100, -100, -97, -95, -92, -89, -88,
                       -87, -86, -86, -87, -91, -100, -110, -113, -100, -69, -20, 33, 80, 105, 103, 75, 30, -17, -57,
                       -82, -92, -91, -84, -76, -68, -64, -65, -70, -78, -88, -97, -102, -104, -104, -102, -101, -98,
                       -92, -89, -87, -85, -85, -85, -88, -96, -106, -113, -110, -88, -47, 5, 58, 96, 108, 92, 53, 3,
                       -43, -78, -96, -97, -89, -76, -65, -57, -55, -59, -68, -79, -90, -98, -102, -102, -99, -95, -92,
                       -89, -87, -87, -87, -88, -89, -89, -92, -99, -108, -112, -102, -74, -28, 25, 73, 101, 101, 74,
                       29, -21, -64, -92, -103, -89, -77, -66, -60, -58, -62, -70, -80, -89, -96, -98, -98, -96, -95,
                       -93, -93, -93, -93, -92, -90, -89, -89, -92, -99, -109, -112, -100, -70, -22, 31, 79, 106, 105,
                       79, 34, -14, -55, -81, -92, -90, -81, -71, -63, -58, -58, -64, -73, -84, -93, -99, -100, -98,
                       -94, -91, -88, -87, -87, -87, -88, -88, -87, -87, -88, -100, -108, -108, -92, -56, -5, 48, 91,
                       110, 100, 66, 19, -28, -65, -87, -93, -89, -80, -70, -63, -60, -62, -68, -77, -86, -94, -98, -98,
                       -95, -91, -89, -87, -87, -88, -89, -89, -89, -88, -88, -89, -95, -103, -108, -102, -79, -39, 9,
                       55, 86, 92, 73, 36, -8, -50, -80, -95, -96, -88, -77, -66, -58, -55, -60, -82, -95, -103, -105,
                       -101, -94, -86, -79, -76, -77, -80, -84, -88, -91, -94, -99, -106, -111, -106, -85, -47, 1, 49,
                       83, 93, 78, 42, -2, -45, -77, -94, -96, -90, -80, -68, -59, -55, -56, -64, -75, -86, -96, -100,
                       -100, -97, -93, -89, -87, -86, -87, -88, -88, -89, -91, -97, -105, -108, -99, -71, -27, 23, 69,
                       96, 82, 40, -6, -48, -75, -88, -88, -82, -73, -65, -60, -59, -63, -71, -81, -90, -96, -99, -99,
                       -96, -93, -90, -87, -86, -86, -87, -87, -88, -89, -93, -101, -110, -111, -97, -64, -15, 39, 84,
                       107, 102, 73, 28, -17, -54, -76, -85, -83, -77, -70, -64, -61, -61, -66, -75, -85, -94, -101,
                       -103, -102, -99, -96, -93, -91, -90, -90, -89, -88, -89, -91, -99, -108, -110, -96, -64, -17, 34,
                       77, 98, 92, 62, 19, -25, -61, -83, -90, -86, -78, -68, -60, -55, -55, -60, -70, -83, -95, -103,
                       -108, -107, -104, -99, -93, -88, -84, -81, -78, -77, -78, -83, -93, -102, -102, -86, -54, -10,
                       34, 68, 82, 73, 45, 7, -30, -59, -77, -83, -81, -73, -56, -53, -57, -71, -90, -109, -123, -127,
                       -121, -108, -91, -76, -68, -66, -70, -76, -82, -88, -96, -103, -105, -95, -69, -30, 15, 56, 82,
                       86, 68, 36, 0, -31, -55, -68, -73, -73, -69, -64, -59, -58, -65, -78, -96, -113, -125, -127,
                       -120, -106, -88, -73, -64, -62, -67, -76, -86, -98, -108, -111, -101, -76, -37, 7, 48, 85, 70,
                       40, 4, -27, -51, -66, -72, -73, -72, -68, -62, -59, -62, -72, -89, -109, -125, -128, -128, -120,
                       -104, -86, -72, -65, -67, -73, -83, -94, -106, -114, -113, -96, -63, -18, 28, 65, 83, 79, 56, 23,
                       -11, -39, -58, -68, -72, -73, -71, -67, -62, -60, -65, -78, -96, -114, -127, -128, -122, -107,
                       -90, -75, -64, -62, -74, -87, -103, -111, -105, -82, -44, 1, 41, 65, 65, 45, 11, -24, -52, -69,
                       -74, -71, -66, -63, -63, -62, -63, -65, -69, -77, -89, -102, -114, -121, -123, -117, -105, -88,
                       -73, -63, -61, -71, -90, -110, -120, -109, -74, -21, 35, 80, 100, 90, 57, 14, -27, -58, -75, -80,
                       -78, -76, -74, -73, -71, -70, -71, -73, -77, -93, -101, -109, -113, -113, -108, -98, -86, -75,
                       -70, -73, -86, -99, -102, -86, -50, 0, 46, 78, 84, 65, 25, -19, -59, -85, -95, -91, -82, -72,
                       -64, -61, -63, -69, -76, -84, -90, -95, -99, -103, -108, -113, -115, -113, -106, -97, -91, -89,
                       -85, -73, -49, -13, 28, 65, 87, 86, 63, 23, -21, -61, -91, -106, -108, -98, -66, -50, -42, -44,
                       -56, -72, -88, -99, -103, -101, -97, -94, -94, -98, -105, -115, -123, -122, -104, -67, -16, 36,
                       79, 99, 91, 61, 19, -22, -57, -79, -91, -95, -97, -96, -91, -83, -72, -63, -56, -54, -58, -66,
                       -77, -88, -96, -100, -99, -96, -96, -103, -114, -118, -108, -79, -32, 21, 67, 92, 91, 65, 23,
                       -22, -61, -95, -91, -82, -72, -67, -68, -78, -92, -106, -115, -114, -105, -91, -77, -68, -68,
                       -76, -93, -111, -120, -112, -83, -39, 9, 50, 69, 64, 39, 2, -32, -58, -71, -72, -68, -64, -63,
                       -64, -65, -67, -69, -74, -82, -95, -109, -122, -128, -128, -125, -114, -102, -94, -86, -72, -49,
                       -15, 23, 59, 81, 82, 61, 20, -30, -79, -125, -125, -112, -92, -71, -53, -44, -46, -57, -73, -87,
                       -96, -98, -95, -91, -91, -97, -110, -126, -128, -125, -93, -44, 11, 59, 85, 83, 54, 10, -36, -74,
                       -97, -104, -101, -94, -89, -88, -89, -93, -96, -96, -93, -87, -80, -76, -77, -81, -88, -95, -101,
                       -106, -109, -106, -92, -67, -34, 1, 31, 46, 45, 27, 0, -31, -78, -88, -90, -86, -79, -71, -65,
                       -63, -69, -81, -95, -109, -117, -117, -110, -97, -82, -72, -70, -74, -78, -72, -53, -20, 19, 52,
                       67, 58, 27, -16, -61, -95, -113, -114, -102, -83, -65, -53, -48, -53, -65, -79, -91, -97, -97,
                       -94, -92, -93, -98, -106, -117, -126, -126, -109, -73, -24, 28, 71, 92, 86, 57, 13, -30, -88,
                       -97, -97, -95, -92, -90, -88, -85, -82, -78, -74, -72, -72, -75, -81, -90, -96, -99, -100, -100,
                       -104, -106, -103, -88, -59, -19, 21, 54, 68, 61, 36, 0, -40, -72, -92, -100, -96, -87, -75, -68,
                       -67, -74, -86, -101, -111, -115, -110, -99, -86, -76, -69, -70, -77, -92, -106, -114, -106, -81,
                       -42, 1, 37, 56, 53, 4, -35, -65, -82, -84, -78, -70, -64, -62, -63, -66, -71, -78, -87, -97,
                       -108, -117, -124, -126, -122, -112, -98, -86, -81, -81, -80, -72, -50, -13, 29, 66, 85, 78, 46,
                       -4, -58, -102, -127, -128, -115, -89, -62, -39, -28, -31, -44, -63, -82, -94, -99, -99, -97, -95,
                       -97, -102, -110, -122, -128, -128, -120, -87, -37, 19, 97, 99, 73, 30, -16, -58, -87, -102, -105,
                       -103, -99, -93, -87, -82, -77, -74, -74, -76, -81, -88, -95, -101, -103, -102, -98, -96, -100,
                       -107, -111, -103, -76, -31, 22, 69, 96, 94, 67, 21, -27, -69, -94, -104, -100, -91, -81, -74,
                       -71, -72, -77, -84, -92, -97, -101, -102, -102, -99, -96, -92, -88, -86, -91, -99, -104, -75,
                       -34, 16, 62, 91, 93, 70, 27, -21, -64, -94, -106, -103, -91, -77, -65, -58, -58, -64, -74, -85,
                       -94, -99, -100, -99, -97, -94, -92, -89, -89, -93, -100, -105, -99, -75, -35, 13, 60, 90, 95, 74,
                       34, -12, -56, -85, -98, -96, -86, -73, -62, -56, -56, -61, -71, -83, -92, -98, -100, -100, -99,
                       -97, -94, -92, -103, -110, -108, -89, -53, -3, 48, 87, 103, 92, 58, 10, -36, -72, -93, -97, -91,
                       -80, -69, -61, -58, -61, -70, -81, -92, -100, -103, -104, -102, -99, -96, -93, -90, -90, -95,
                       -104, -112, -107, -84, -41, 12, 63, 96, 102, 79, 36, -14, -59, -89, -102, -99, -89, -76, -65,
                       -58, -57, -62, -72, -85, -98, -106, -111, -111, -105, -101, -98, -95, -92, -90, -93, -101, -109,
                       -108, -89, -51, 0, 51, 89, 103, 88, 51, 2, -44, -78, -95, -97, -90, -78, -68, -62, -61, -67, -77,
                       -88, -98, -104, -106, -104, -102, -99, -97, -96, -96, -96, -95, -96, -101, -109, -115, -110, -87,
                       -45, 7, 59, 95, 106, 88, 48, -3, -51, -86, -103, -104, -95, -81, -68, -58, -64, -74, -86, -96,
                       -103, -105, -104, -101, -98, -95, -93, -90, -89, -89, -92, -102, -111, -112, -96, -60, -9, 45,
                       89, 109, 99, 64, 12, -39, -80, -104, -108, -99, -82, -65, -52, -45, -47, -55, -67, -82, -94,
                       -103, -107, -108, -107, -103, -99, -94, -89, -87, -91, -100, -107, -104, -83, -43, 8, 59, 96,
                       107, 89, 50, -42, -75, -91, -92, -85, -76, -69, -64, -62, -66, -73, -82, -92, -98, -101, -99,
                       -97, -93, -92, -91, -92, -94, -95, -97, -102, -108, -109, -95, -65, -19, 33, 77, 102, 100, 72,
                       29, -18, -57, -83, -92, -90, -82, -72, -63, -57, -56, -60, -69, -81, -93, -102, -106, -106, -102,
                       -95, -88, -83, -80, -81, -84, -87, -91, -96, -112, -115, -104, -73, -25, 29, 76, 102, 101, 73,
                       29, -18, -58, -83, -93, -91, -83, -73, -65, -59, -58, -62, -71, -82, -94, -101, -105, -105, -102,
                       -98, -94, -90, -87, -85, -83, -83, -85, -93, -104, -111, -105, -81, -39, 12, 59, 89, 93, 71, 32,
                       -13, -54, -82, -95, -95, -87, -76, -66, -59, -58, -62, -70, -79, -89, -100, -102, -103, -104,
                       -104, -102, -99, -94, -89, -85, -87, -95, -101, -98, -79, -40, 9, 59, 93, 100, 79, 37, -12, -56,
                       -84, -95, -91, -79, -68, -59, -55, -56, -60, -68, -75, -83, -90, -97, -103, -110, -115, -118,
                       -117, -112, -103, -91, -79, -71, -72, -82, -95, -100, -89, -56, -7, 45, 85, 99, 86, 49, 1, -42,
                       -73, -86, -78, -68, -61, -57, -57, -61, -67, -76, -85, -94, -101, -106, -108, -108, -105, -100,
                       -94, -87, -82, -80, -80, -86, -97, -111, -121, -116, -92, -49, 5, 57, 91, 99, 80, 41, -4, -46,
                       -75, -88, -89, -83, -74, -66, -61, -60, -64, -73, -84, -95, -103, -106, -106, -103, -99, -96,
                       -92, -90, -88, -86, -85, -84, -84, -89]

    print(get_nn_intervals_ms(pulse_wave_data))
    print(get_data_metrics(pulse_wave_data))
