"""Simple Travelling Salesperson Problem (TSP) between cities."""
#Before running this program please run below mentioned command without adding # to terminal or CMD
#pip install ortools


from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = [
       [0, 291, 842, 229, 1574, 1278, 1349, 2117, 138, 2384, 508, 1591, 715, 513, 118, 1626, 1017, 258, 660, 230, 82, 1881, 674, 1146, 1003, 547, 1185, 1135, 790, 985, 125, 825, 1360, 2386, 2159, 1331, 132, 541, 403, 2485, 975, 2423, 1953, 1777, 847, 1024, 1168, 176, 310], [291, 0, 568, 505, 1344, 1019, 1336, 2065, 367, 2298, 279, 1498, 426, 353, 389, 1516, 918, 53, 455, 91, 277, 1858, 519, 894, 843, 494, 1076, 919, 728, 714, 417, 595, 1177, 2373, 2007, 1280, 331, 250, 111, 2398, 945, 2392, 1867, 1626, 704, 736, 1027, 125, 28], [842, 568, 0, 1067, 817, 458, 1244, 1831, 935, 1987, 352, 1227, 193, 435, 951, 1209, 738, 617, 307, 611, 844, 1693, 427, 350, 547, 620, 832, 456, 698, 145, 961, 215, 754, 2196, 1589, 1123, 900, 355, 469, 2081, 902, 2179, 1580, 1228, 509, 228, 725, 693, 542], [229, 505, 1067, 0, 1802, 1507, 1499, 2277, 149, 2560, 738, 1777, 931, 739, 116, 1819, 1214, 462, 888, 456, 228, 2028, 895, 1375, 1220, 746, 1384, 1362, 980, 1212, 138, 1055, 1580, 2521, 2366, 1501, 189, 751, 614, 2662, 1143, 2570, 2133, 1985, 1062, 1241, 1378, 380, 528], [1574, 1344, 817, 1802, 0, 390, 1154, 1368, 1692, 1388, 1073, 848, 1009, 1065, 1692, 771, 814, 1397, 914, 1359, 1600, 1361, 925, 467, 655, 1127, 729, 442, 996, 690, 1676, 751, 360, 1770, 869, 983, 1663, 1167, 1264, 1464, 1044, 1703, 1073, 603, 800, 801, 617, 1460, 1316], [1278, 1019, 458, 1507, 390, 0, 1241, 1641, 1382, 1720, 769, 1059, 641, 802, 1392, 1007, 780, 1070, 643, 1051, 1290, 1574, 702, 140, 563, 923, 778, 303, 877, 317, 1391, 469, 504, 2035, 1240, 1082, 1348, 812, 925, 1804, 1006, 1988, 1358, 925, 651, 411, 645, 1141, 992], [1349, 1336, 1244, 1499, 1154, 1241, 0, 780, 1484, 1091, 1133, 440, 1358, 1007, 1441, 532, 506, 1374, 1002, 1274, 1424, 532, 879, 1148, 716, 842, 462, 943, 613, 1256, 1368, 1029, 800, 1040, 1097, 170, 1482, 1371, 1349, 1190, 390, 1074, 692, 814, 739, 1433, 595, 1372, 1315], [2117, 2065, 1831, 2277, 1368, 1641, 780, 0, 2254, 358, 1826, 604, 1988, 1716, 2213, 636, 1153, 2109, 1667, 2015, 2187, 289, 1563, 1608, 1293, 1585, 1013, 1412, 1341, 1790, 2144, 1631, 1139, 402, 736, 788, 2248, 2049, 2057, 443, 1142, 349, 295, 772, 1387, 1964, 1106, 2120, 2042], [138, 367, 935, 149, 1692, 1382, 1484, 2254, 0, 2522, 620, 1730, 790, 641, 63, 1764, 1155, 321, 781, 333, 92, 2016, 805, 1254, 1136, 686, 1323, 1256, 929, 1081, 152, 941, 1490, 2519, 2295, 1469, 40, 609, 473, 2624, 1112, 2558, 2092, 1913, 982, 1100, 1304, 242, 392], [2384, 2298, 1987, 2560, 1388, 1720, 1091, 358, 2522, 0, 2041, 799, 2161, 1944, 2488, 788, 1381, 2345, 1871, 2258, 2450, 644, 1781, 1716, 1478, 1840, 1222, 1540, 1594, 1921, 2424, 1805, 1245, 566, 581, 1060, 2512, 2247, 2275, 101, 1417, 432, 431, 794, 1598, 2086, 1287, 2366, 2272], [508, 279, 352, 738, 1073, 769, 1133, 1826, 620, 2041, 0, 1244, 301, 134, 624, 1254, 672, 333, 177, 286, 527, 1638, 263, 637, 571, 340, 820, 641, 521, 488, 621, 321, 898, 2153, 1730, 1053, 590, 240, 234, 2140, 747, 2161, 1613, 1350, 442, 566, 759, 388, 251], [1591, 1498, 1227, 1777, 848, 1059, 440, 604, 1730, 799, 1244, 0, 1387, 1145, 1698, 97, 581, 1545, 1077, 1458, 1654, 519, 983, 1012, 692, 1044, 423, 811, 801, 1186, 1640, 1029, 555, 982, 664, 317, 1718, 1457, 1477, 900, 646, 953, 369, 381, 801, 1362, 503, 1567, 1473], [715, 426, 193, 931, 1009, 641, 1358, 1988, 790, 2161, 301, 1387, 0, 425, 815, 1376, 862, 469, 359, 489, 702, 1832, 488, 541, 695, 632, 977, 643, 771, 324, 840, 357, 939, 2342, 1778, 1252, 752, 183, 316, 2256, 991, 2333, 1746, 1413, 621, 310, 883, 550, 404], [513, 353, 435, 739, 1065, 802, 1007, 1716, 641, 1944, 134, 1145, 425, 0, 632, 1163, 565, 402, 158, 324, 552, 1519, 168, 663, 499, 207, 722, 623, 393, 555, 610, 333, 849, 2034, 1664, 936, 617, 373, 343, 2044, 619, 2047, 1514, 1282, 352, 661, 678, 435, 328], [118, 389, 951, 116, 1692, 1392, 1441, 2213, 63, 2488, 624, 1698, 815, 632, 0, 1735, 1127, 347, 778, 341, 115, 1972, 791, 1261, 1120, 658, 1297, 1253, 897, 1096, 89, 943, 1478, 2472, 2274, 1431, 95, 637, 499, 2589, 1073, 2514, 2058, 1892, 964, 1126, 1283, 265, 412], [1626, 1516, 1209, 1819, 771, 1007, 532, 636, 1764, 788, 1254, 97, 1376, 1163, 1735, 0, 609, 1565, 1083, 1482, 1686, 590, 997, 971, 690, 1078, 442, 777, 839, 1157, 1681, 1019, 503, 1027, 586, 398, 1750, 1459, 1488, 886, 704, 986, 370, 285, 812, 1330, 498, 1591, 1490], [1017, 918, 738, 1214, 814, 780, 506, 1153, 1155, 1381, 672, 581, 862, 565, 1127, 609, 0, 964, 518, 876, 1076, 973, 409, 668, 221, 469, 169, 478, 238, 752, 1076, 524, 470, 1486, 1155, 390, 1141, 899, 903, 1482, 233, 1489, 950, 779, 241, 928, 206, 985, 893], [258, 53, 617, 462, 1397, 1070, 1374, 2109, 321, 2345, 333, 1545, 469, 402, 347, 1565, 964, 0, 508, 104, 233, 1899, 568, 947, 895, 531, 1124, 972, 769, 763, 383, 649, 1230, 2412, 2059, 1323, 283, 289, 152, 2445, 984, 2434, 1914, 1678, 754, 780, 1078, 83, 82], [660, 455, 307, 888, 914, 643, 1002, 1667, 781, 1871, 177, 1077, 359, 158, 778, 1083, 518, 508, 0, 451, 690, 1492, 129, 504, 395, 314, 655, 475, 412, 409, 763, 174, 721, 2005, 1553, 907, 754, 382, 408, 1969, 631, 2006, 1445, 1173, 279, 536, 585, 558, 427], [230, 91, 611, 456, 1359, 1051, 1274, 2015, 333, 2258, 286, 1458, 489, 324, 341, 1482, 876, 104, 451, 0, 241, 1800, 492, 921, 823, 432, 1039, 925, 673, 755, 352, 607, 1169, 2313, 1989, 1227, 303, 324, 190, 2359, 884, 2337, 1827, 1606, 676, 796, 1002, 110, 95], [82, 277, 844, 228, 1600, 1290, 1424, 2187, 92, 2450, 527, 1654, 702, 552, 115, 1686, 1076, 233, 690, 241, 0, 1956, 717, 1161, 1049, 610, 1244, 1165, 855, 989, 172, 848, 1402, 2462, 2211, 1400, 65, 522, 385, 2551, 1046, 2497, 2018, 1828, 896, 1013, 1220, 152, 302], [1881, 1858, 1693, 2028, 1361, 1574, 532, 289, 2016, 644, 1638, 519, 1832, 1519, 1972, 590, 973, 1899, 1492, 1800, 1956, 0, 1377, 1517, 1146, 1368, 860, 1311, 1130, 1673, 1898, 1483, 1073, 515, 907, 585, 2014, 1871, 1862, 732, 916, 542, 401, 816, 1214, 1852, 975, 1901, 1837], [674, 519, 427, 895, 925, 702, 879, 1563, 805, 1781, 263, 983, 488, 168, 791, 997, 409, 568, 129, 492, 717, 1377, 0, 562, 332, 221, 559, 483, 282, 510, 761, 251, 686, 1892, 1496, 792, 783, 494, 495, 1881, 503, 1898, 1352, 1114, 185, 653, 510, 603, 492], [1146, 894, 350, 1375, 467, 140, 1148, 1608, 1254, 1716, 637, 1012, 541, 663, 1261, 971, 668, 947, 504, 921, 1161, 1517, 562, 0, 447, 783, 689, 207, 745, 226, 1257, 329, 473, 1994, 1271, 997, 1221, 702, 806, 1805, 887, 1957, 1334, 931, 518, 371, 558, 1015, 867], [1003, 843, 547, 1220, 655, 563, 716, 1293, 1136, 1478, 571, 692, 695, 499, 1120, 690, 221, 895, 395, 823, 1049, 1146, 332, 447, 0, 491, 285, 264, 342, 540, 1084, 339, 363, 1650, 1165, 581, 1115, 768, 804, 1576, 443, 1638, 1056, 783, 158, 719, 191, 934, 816], [547, 494, 620, 746, 1127, 923, 842, 1585, 686, 1840, 340, 1044, 632, 207, 658, 1078, 469, 531, 314, 432, 610, 1368, 221, 783, 491, 0, 639, 690, 245, 721, 608, 468, 853, 1881, 1619, 797, 673, 574, 519, 1941, 452, 1905, 1408, 1239, 333, 849, 633, 535, 475], [1185, 1076, 832, 1384, 729, 778, 462, 1013, 1323, 1222, 820, 423, 977, 722, 1297, 442, 169, 1124, 655, 1039, 1244, 860, 559, 689, 285, 639, 0, 482, 406, 818, 1245, 624, 369, 1365, 986, 307, 1308, 1036, 1053, 1322, 339, 1356, 793, 613, 377, 997, 133, 1149, 1050], [1135, 919, 456, 1362, 442, 303, 943, 1412, 1256, 1540, 641, 811, 643, 623, 1253, 777, 478, 972, 475, 925, 1165, 1311, 483, 207, 264, 690, 482, 0, 595, 381, 1234, 324, 299, 1794, 1134, 790, 1228, 772, 849, 1632, 706, 1762, 1145, 772, 377, 552, 351, 1029, 890], [790, 728, 698, 980, 996, 877, 613, 1341, 929, 1594, 521, 801, 771, 393, 897, 839, 238, 769, 412, 673, 855, 1130, 282, 745, 342, 245, 406, 595, 0, 757, 842, 496, 679, 1645, 1393, 554, 918, 762, 735, 1696, 225, 1664, 1163, 1018, 226, 916, 429, 778, 706], [985, 714, 145, 1212, 690, 317, 1256, 1790, 1081, 1921, 488, 1186, 324, 555, 1096, 1157, 752, 763, 409, 755, 989, 1673, 510, 226, 540, 721, 818, 381, 757, 0, 1104, 263, 676, 2166, 1493, 1121, 1045, 497, 615, 2012, 942, 2139, 1526, 1145, 545, 179, 698, 838, 688], [125, 417, 961, 138, 1676, 1391, 1368, 2144, 152, 2424, 621, 1640, 840, 610, 89, 1681, 1076, 383, 763, 352, 172, 1898, 761, 1257, 1084, 608, 1245, 1234, 842, 1104, 0, 932, 1445, 2395, 2228, 1365, 177, 667, 528, 2526, 1007, 2440, 1996, 1847, 926, 1148, 1240, 300, 435], [825, 595, 215, 1055, 751, 469, 1029, 1631, 941, 1805, 321, 1029, 357, 333, 943, 1019, 524, 649, 174, 607, 848, 1483, 251, 329, 339, 468, 624, 324, 496, 263, 932, 0, 601, 1989, 1443, 910, 911, 455, 525, 1901, 691, 1977, 1389, 1070, 295, 421, 525, 709, 567], [1360, 1177, 754, 1580, 360, 504, 800, 1139, 1490, 1245, 898, 555, 939, 849, 1478, 503, 470, 1230, 721, 1169, 1402, 1073, 686, 473, 363, 853, 369, 299, 679, 676, 1445, 601, 0, 1530, 842, 631, 1467, 1057, 1121, 1336, 693, 1487, 861, 473, 521, 841, 265, 1278, 1149], [2386, 2373, 2196, 2521, 1770, 2035, 1040, 402, 2519, 566, 2153, 982, 2342, 2034, 2472, 1027, 1486, 2412, 2005, 2313, 2462, 515, 1892, 1994, 1650, 1881, 1365, 1794, 1645, 2166, 2395, 1989, 1530, 0, 1095, 1099, 2518, 2385, 2378, 588, 1428, 144, 697, 1173, 1727, 2343, 1471, 2413, 2351], [2159, 2007, 1589, 2366, 869, 1240, 1097, 736, 2295, 581, 1730, 664, 1778, 1664, 2274, 586, 1155, 2059, 1553, 1989, 2211, 907, 1496, 1271, 1165, 1619, 986, 1134, 1393, 1493, 2228, 1443, 842, 1095, 0, 981, 2276, 1899, 1958, 632, 1285, 984, 506, 382, 1315, 1638, 991, 2099, 1979], [1331, 1280, 1123, 1501, 983, 1082, 170, 788, 1469, 1060, 1053, 317, 1252, 936, 1431, 398, 390, 1323, 907, 1227, 1400, 585, 792, 997, 581, 797, 307, 790, 554, 1121, 1365, 910, 631, 1099, 981, 0, 1462, 1286, 1279, 1161, 358, 1111, 636, 668, 631, 1300, 439, 1332, 1258], [132, 331, 900, 189, 1663, 1348, 1482, 2248, 40, 2512, 590, 1718, 752, 617, 95, 1750, 1141, 283, 754, 303, 65, 2014, 783, 1221, 1115, 673, 1308, 1228, 918, 1045, 177, 911, 1467, 2518, 2276, 1462, 0, 570, 436, 2614, 1106, 2556, 2081, 1894, 962, 1062, 1285, 207, 357], [541, 250, 355, 751, 1167, 812, 1371, 2049, 609, 2247, 240, 1457, 183, 373, 637, 1459, 899, 289, 382, 324, 522, 1871, 494, 702, 768, 574, 1036, 772, 762, 497, 667, 455, 1057, 2385, 1899, 1286, 570, 0, 138, 2345, 987, 2388, 1824, 1525, 661, 492, 960, 371, 232], [403, 111, 469, 614, 1264, 925, 1349, 2057, 473, 2275, 234, 1477, 316, 343, 499, 1488, 903, 152, 408, 190, 385, 1862, 495, 806, 804, 519, 1053, 849, 735, 615, 528, 525, 1121, 2378, 1958, 1279, 436, 138, 0, 2374, 959, 2390, 1847, 1579, 676, 627, 992, 234, 96], [2485, 2398, 2081, 2662, 1464, 1804, 1190, 443, 2624, 101, 2140, 900, 2256, 2044, 2589, 886, 1482, 2445, 1969, 2359, 2551, 732, 1881, 1805, 1576, 1941, 1322, 1632, 1696, 2012, 2526, 1901, 1336, 588, 632, 1161, 2614, 2345, 2374, 0, 1518, 446, 532, 878, 1697, 2175, 1385, 2467, 2372], [975, 945, 902, 1143, 1044, 1006, 390, 1142, 1112, 1417, 747, 646, 991, 619, 1073, 704, 233, 984, 631, 884, 1046, 916, 503, 887, 443, 452, 339, 706, 225, 942, 1007, 691, 693, 1428, 1285, 358, 1106, 987, 959, 1518, 0, 1454, 991, 929, 397, 1111, 428, 985, 924], [2423, 2392, 2179, 2570, 1703, 1988, 1074, 349, 2558, 432, 2161, 953, 2333, 2047, 2514, 986, 1489, 2434, 2006, 2337, 2497, 542, 1898, 1957, 1638, 1905, 1356, 1762, 1664, 2139, 2440, 1977, 1487, 144, 984, 1111, 2556, 2388, 2390, 446, 1454, 0, 632, 1100, 1727, 2314, 1454, 2439, 2369], [1953, 1867, 1580, 2133, 1073, 1358, 692, 295, 2092, 431, 1613, 369, 1746, 1514, 2058, 370, 950, 1914, 1445, 1827, 2018, 401, 1352, 1334, 1056, 1408, 793, 1145, 1163, 1526, 1996, 1389, 861, 697, 506, 636, 2081, 1824, 1847, 532, 991, 632, 0, 477, 1170, 1697, 865, 1935, 1842], [1777, 1626, 1228, 1985, 603, 925, 814, 772, 1913, 794, 1350, 381, 1413, 1282, 1892, 285, 779, 1678, 1173, 1606, 1828, 816, 1114, 931, 783, 1239, 613, 772, 1018, 1145, 1847, 1070, 473, 1173, 382, 668, 1894, 1525, 1579, 878, 929, 1100, 477, 0, 932, 1302, 609, 1717, 1598], [847, 704, 509, 1062, 800, 651, 739, 1387, 982, 1598, 442, 801, 621, 352, 964, 812, 241, 754, 279, 676, 896, 1214, 185, 518, 158, 333, 377, 377, 226, 545, 926, 295, 521, 1727, 1315, 631, 962, 661, 676, 1697, 397, 1727, 1170, 932, 0, 713, 325, 787, 677], [1024, 736, 228, 1241, 801, 411, 1433, 1964, 1100, 2086, 566, 1362, 310, 661, 1126, 1330, 928, 780, 536, 796, 1013, 1852, 653, 371, 719, 849, 997, 552, 916, 179, 1148, 421, 841, 2343, 1638, 1300, 1062, 492, 627, 2175, 1111, 2314, 1697, 1302, 713, 0, 877, 861, 714], [1168, 1027, 725, 1378, 617, 645, 595, 1106, 1304, 1287, 759, 503, 883, 678, 1283, 498, 206, 1078, 585, 1002, 1220, 975, 510, 558, 191, 633, 133, 351, 429, 698, 1240, 525, 265, 1471, 991, 439, 1285, 960, 992, 1385, 428, 1454, 865, 609, 325, 877, 0, 1112, 1000], [176, 125, 693, 380, 1460, 1141, 1372, 2120, 242, 2366, 388, 1567, 550, 435, 265, 1591, 985, 83, 558, 110, 152, 1901, 603, 1015, 934, 535, 1149, 1029, 778, 838, 300, 709, 1278, 2413, 2099, 1332, 207, 371, 234, 2467, 985, 2439, 1935, 1717, 787, 861, 1112, 0, 150], [310, 28, 542, 528, 1316, 992, 1315, 2042, 392, 2272, 251, 1473, 404, 328, 412, 1490, 893, 82, 427, 95, 302, 1837, 492, 867, 816, 475, 1050, 890, 706, 688, 435, 567, 1149, 2351, 1979, 1258, 357, 232, 96, 2372, 924, 2369, 1842, 1598, 677, 714, 1000, 150, 0]
    ]  # yapf: disable
    data['num_vehicles'] = 1
    data['depot'] = 20
    return data


def print_solution(manager, routing, solution):
    """Prints solution on console."""
    print('Objective: {} miles'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'Route for vehicle 0:\n'
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    print(plan_output)
    plan_output += 'Route distance: {}miles\n'.format(route_distance)


def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(manager, routing, solution)


if __name__ == '__main__':
    main()