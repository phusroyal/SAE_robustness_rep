=== CORRECTED HIDDEN SPACE POLYTOPE ANALYSIS ===
ISSUE FOUND: Previous implementation tracked SAE boundary crossings
CORRECTION: Now tracking ReLU polytopes boundaries in HIDDEN space

Source: 'The film explores love and trauma through non-linear storytelling, blending...'
Target: 'The film explores love and trauma through non-linear storytelling, blending magical realism with emotionally...'

Source hidden shape: torch.Size([2304])
Target hidden shape: torch.Size([2304])
Source polytope active dims: 1142.0/2304
Target polytope active dims: 1125.0/2304
Polytope Hamming distance: 595

Running CORRECTED hidden polytope transition analysis...
Step 0: loss = 11.902927, ||δ|| = 0.000000, boundaries = 0, hamming = 595
Step 20: Hidden polytope boundary #19
  Deactivated dims: [798, 1304, 1306]...
Step 40: Hidden polytope boundary #37
  Deactivated dims: [865, 1271, 1608, 2115, 2232]...
Step 50: loss = 9.516148, ||δ|| = 22.046921, boundaries = 45, hamming = 480
Step 80: Hidden polytope boundary #68
  Activated dims: [131]...
  Deactivated dims: [2196]...
Step 100: Hidden polytope boundary #86
  Deactivated dims: [2017, 2107]...
Step 100: loss = 7.656289, ||δ|| = 40.243969, boundaries = 86, hamming = 398
Step 120: Hidden polytope boundary #101
  Activated dims: [1027]...
  Deactivated dims: [1478]...
Step 150: loss = 6.199106, ||δ|| = 55.715729, boundaries = 123, hamming = 314
Step 160: Hidden polytope boundary #132
  Activated dims: [460]...
  Deactivated dims: [1429]...
Step 200: loss = 5.047878, ||δ|| = 69.056694, boundaries = 157, hamming = 262
Step 220: Hidden polytope boundary #171
  Deactivated dims: [433]...
Step 240: Hidden polytope boundary #182
  Activated dims: [640]...
Step 250: loss = 4.132979, ||δ|| = 80.629021, boundaries = 187, hamming = 213
Step 280: Hidden polytope boundary #204
  Deactivated dims: [1073]...
Step 300: loss = 3.402362, ||δ|| = 90.701904, boundaries = 216, hamming = 176
Step 320: Hidden polytope boundary #226
  Activated dims: [2084]...
Step 340: Hidden polytope boundary #235
  Deactivated dims: [1631]...
Step 350: loss = 2.816348, ||δ|| = 99.488976, boundaries = 238, hamming = 147
Step 400: loss = 2.344322, ||δ|| = 107.165794, boundaries = 263, hamming = 114
Step 450: loss = 1.962486, ||δ|| = 113.879715, boundaries = 278, hamming = 97
Step 480: Hidden polytope boundary #287
  Deactivated dims: [1419]...
Step 500: loss = 1.652253, ||δ|| = 119.756416, boundaries = 292, hamming = 80
Step 550: loss = 1.399052, ||δ|| = 124.904045, boundaries = 304, hamming = 65
Step 560: Hidden polytope boundary #306
  Activated dims: [1297]...
Step 600: loss = 1.191430, ||δ|| = 129.416321, boundaries = 312, hamming = 56
Step 620: Hidden polytope boundary #319
  Deactivated dims: [2298]...
Step 650: loss = 1.020358, ||δ|| = 133.374634, boundaries = 328, hamming = 39
Step 700: Hidden polytope boundary #334
  Activated dims: [12]...
Step 700: loss = 0.878698, ||δ|| = 136.849762, boundaries = 334, hamming = 33
Step 750: loss = 0.760792, ||δ|| = 139.903290, boundaries = 341, hamming = 25
Step 800: loss = 0.662142, ||δ|| = 142.588852, boundaries = 344, hamming = 22
Step 850: loss = 0.579162, ||δ|| = 144.953171, boundaries = 348, hamming = 18
Step 900: loss = 0.508986, ||δ|| = 147.036957, boundaries = 350, hamming = 16
Step 950: loss = 0.449316, ||δ|| = 148.875595, boundaries = 356, hamming = 10
Step 1000: loss = 0.398305, ||δ|| = 150.500031, boundaries = 357, hamming = 9
Step 1050: loss = 0.354461, ||δ|| = 151.937164, boundaries = 358, hamming = 8
Step 1100: loss = 0.316579, ||δ|| = 153.210464, boundaries = 358, hamming = 8
Step 1150: loss = 0.283679, ||δ|| = 154.340363, boundaries = 359, hamming = 7
Step 1200: loss = 0.254961, ||δ|| = 155.344666, boundaries = 360, hamming = 6
Step 1250: loss = 0.229773, ||δ|| = 156.238953, boundaries = 360, hamming = 6
Step 1300: loss = 0.207579, ||δ|| = 157.036697, boundaries = 360, hamming = 6
Step 1350: loss = 0.187936, ||δ|| = 157.749771, boundaries = 362, hamming = 4
Step 1400: loss = 0.170479, ||δ|| = 158.388443, boundaries = 362, hamming = 4
Step 1450: loss = 0.154903, ||δ|| = 158.961685, boundaries = 363, hamming = 3
Step 1500: loss = 0.140957, ||δ|| = 159.477325, boundaries = 363, hamming = 3
Step 1550: loss = 0.128427, ||δ|| = 159.942245, boundaries = 363, hamming = 3
Step 1600: loss = 0.117134, ||δ|| = 160.362366, boundaries = 363, hamming = 3
Step 1650: loss = 0.106928, ||δ|| = 160.742935, boundaries = 364, hamming = 2
Step 1700: loss = 0.097681, ||δ|| = 161.088486, boundaries = 364, hamming = 2
Step 1750: loss = 0.089282, ||δ|| = 161.403015, boundaries = 364, hamming = 2
Step 1800: loss = 0.081639, ||δ|| = 161.690018, boundaries = 364, hamming = 2
Step 1850: loss = 0.074670, ||δ|| = 161.952530, boundaries = 364, hamming = 2
Step 1900: loss = 0.068306, ||δ|| = 162.193237, boundaries = 364, hamming = 2
Step 1950: loss = 0.062485, ||δ|| = 162.414474, boundaries = 364, hamming = 2
Step 2000: loss = 0.057156, ||δ|| = 162.618301, boundaries = 364, hamming = 2
Step 2050: loss = 0.052271, ||δ|| = 162.806519, boundaries = 364, hamming = 2
Step 2100: loss = 0.047790, ||δ|| = 162.980713, boundaries = 364, hamming = 2
Step 2150: loss = 0.043677, ||δ|| = 163.142288, boundaries = 364, hamming = 2
Step 2200: loss = 0.039899, ||δ|| = 163.292465, boundaries = 364, hamming = 2
Step 2250: loss = 0.036427, ||δ|| = 163.432312, boundaries = 364, hamming = 2
Step 2300: loss = 0.033237, ||δ|| = 163.562820, boundaries = 364, hamming = 2
Step 2350: loss = 0.030305, ||δ|| = 163.684784, boundaries = 364, hamming = 2
Step 2400: loss = 0.027609, ||δ|| = 163.798996, boundaries = 364, hamming = 2
Step 2450: loss = 0.025132, ||δ|| = 163.906082, boundaries = 365, hamming = 1
Step 2500: loss = 0.022856, ||δ|| = 164.006653, boundaries = 365, hamming = 1
Step 2550: loss = 0.020765, ||δ|| = 164.101212, boundaries = 365, hamming = 1
Step 2600: loss = 0.018846, ||δ|| = 164.190247, boundaries = 365, hamming = 1
Step 2650: loss = 0.017084, ||δ|| = 164.274155, boundaries = 365, hamming = 1
Step 2700: loss = 0.015469, ||δ|| = 164.353333, boundaries = 365, hamming = 1
Step 2750: loss = 0.013989, ||δ|| = 164.428070, boundaries = 365, hamming = 1
Step 2800: loss = 0.012633, ||δ|| = 164.498688, boundaries = 365, hamming = 1
Step 2850: loss = 0.011392, ||δ|| = 164.565460, boundaries = 365, hamming = 1
Step 2900: loss = 0.010259, ||δ|| = 164.628616, boundaries = 365, hamming = 1
Step 2950: loss = 0.009223, ||δ|| = 164.688385, boundaries = 365, hamming = 1
Step 3000: loss = 0.008279, ||δ|| = 164.744949, boundaries = 365, hamming = 1
Step 3050: loss = 0.007418, ||δ|| = 164.798523, boundaries = 366, hamming = 0
Step 3100: loss = 0.006635, ||δ|| = 164.849228, boundaries = 366, hamming = 0
Step 3150: loss = 0.005924, ||δ|| = 164.897247, boundaries = 366, hamming = 0
Step 3200: loss = 0.005278, ||δ|| = 164.942719, boundaries = 366, hamming = 0
Step 3250: loss = 0.004693, ||δ|| = 164.985764, boundaries = 366, hamming = 0
Step 3300: loss = 0.004164, ||δ|| = 165.026489, boundaries = 366, hamming = 0
Step 3350: loss = 0.003686, ||δ|| = 165.065033, boundaries = 366, hamming = 0
Step 3400: loss = 0.003256, ||δ|| = 165.101486, boundaries = 366, hamming = 0
Step 3450: loss = 0.002869, ||δ|| = 165.135941, boundaries = 366, hamming = 0
Step 3500: loss = 0.002521, ||δ|| = 165.168503, boundaries = 366, hamming = 0
Step 3550: loss = 0.002210, ||δ|| = 165.199234, boundaries = 366, hamming = 0
Step 3600: loss = 0.001932, ||δ|| = 165.228241, boundaries = 366, hamming = 0
Step 3650: loss = 0.001683, ||δ|| = 165.255585, boundaries = 366, hamming = 0
Step 3700: loss = 0.001463, ||δ|| = 165.281342, boundaries = 366, hamming = 0
Step 3750: loss = 0.001267, ||δ|| = 165.305588, boundaries = 366, hamming = 0
Step 3800: loss = 0.001094, ||δ|| = 165.328384, boundaries = 366, hamming = 0
Step 3850: loss = 0.000942, ||δ|| = 165.349792, boundaries = 366, hamming = 0
Step 3900: loss = 0.000808, ||δ|| = 165.369888, boundaries = 366, hamming = 0
Step 3950: loss = 0.000690, ||δ|| = 165.388733, boundaries = 366, hamming = 0
Step 4000: loss = 0.000588, ||δ|| = 165.406372, boundaries = 366, hamming = 0
Step 4050: loss = 0.000499, ||δ|| = 165.422852, boundaries = 366, hamming = 0
Step 4100: loss = 0.000421, ||δ|| = 165.438248, boundaries = 366, hamming = 0
Step 4150: loss = 0.000354, ||δ|| = 165.452606, boundaries = 366, hamming = 0
Step 4200: loss = 0.000297, ||δ|| = 165.465958, boundaries = 366, hamming = 0
Step 4250: loss = 0.000248, ||δ|| = 165.478378, boundaries = 366, hamming = 0
Step 4300: loss = 0.000206, ||δ|| = 165.489899, boundaries = 366, hamming = 0
Step 4350: loss = 0.000170, ||δ|| = 165.500549, boundaries = 366, hamming = 0
Step 4400: loss = 0.000140, ||δ|| = 165.510422, boundaries = 366, hamming = 0
Step 4450: loss = 0.000115, ||δ|| = 165.519516, boundaries = 366, hamming = 0
Step 4500: loss = 0.000093, ||δ|| = 165.527893, boundaries = 366, hamming = 0
Step 4550: loss = 0.000076, ||δ|| = 165.535583, boundaries = 366, hamming = 0
Step 4600: loss = 0.000061, ||δ|| = 165.542633, boundaries = 366, hamming = 0
Step 4650: loss = 0.000049, ||δ|| = 165.549103, boundaries = 366, hamming = 0
Step 4700: loss = 0.000039, ||δ|| = 165.554993, boundaries = 366, hamming = 0
Step 4750: loss = 0.000031, ||δ|| = 165.560349, boundaries = 366, hamming = 0
Step 4800: loss = 0.000024, ||δ|| = 165.565216, boundaries = 366, hamming = 0
Step 4850: loss = 0.000019, ||δ|| = 165.569626, boundaries = 366, hamming = 0
Step 4900: loss = 0.000015, ||δ|| = 165.573608, boundaries = 366, hamming = 0
Step 4950: loss = 0.000011, ||δ|| = 165.577179, boundaries = 366, hamming = 0
Step 5000: loss = 0.000009, ||δ|| = 165.580399, boundaries = 366, hamming = 0
Step 5050: loss = 0.000007, ||δ|| = 165.583267, boundaries = 366, hamming = 0
Step 5100: loss = 0.000005, ||δ|| = 165.585846, boundaries = 366, hamming = 0
Step 5150: loss = 0.000004, ||δ|| = 165.588104, boundaries = 366, hamming = 0
Step 5200: loss = 0.000003, ||δ|| = 165.590134, boundaries = 366, hamming = 0
Step 5250: loss = 0.000002, ||δ|| = 165.591919, boundaries = 366, hamming = 0
Step 5300: loss = 0.000002, ||δ|| = 165.593475, boundaries = 366, hamming = 0
Step 5350: loss = 0.000001, ||δ|| = 165.594864, boundaries = 366, hamming = 0
✅ Target hidden reached at step 5368
Hamming distance to target ReLU support: 0

=== CORRECTED RESULTS ===
Minimal perturbation: 165.595306
Hidden polytope boundaries crossed: 366
Target reached: True
Final loss: 0.000001
Optimization steps: 5368