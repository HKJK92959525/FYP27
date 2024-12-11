import numpy as np
import random
import matplotlib.pyplot as plt

# �̶���������
I = 80  # ������������ֶ�ָ��
J = 40  # ҽԺ����
M = 150 # ��Ԥ��
C = 10  # ����ƽ̨�������
T = 9   # �����Ӧʱ�䣨���ӣ�
V = 1 # ���˻��ٶ�

# �̶��ľ�������������
# ����㵽��ʩ��ľ������ (I x J)
dij = [
  [714.88, 722.4, 610.36, 518.54, 547.03, 689.7, 525.68, 626.83, 553.39, 533.77, 245.05, 694.5, 534.12, 214.0, 634.85, 432.2, 631.05, 637.13, 585.97, 645.89, 154.64, 131.24, 347.99, 168.51, 415.93, 746.78, 193.75, 150.88, 319.86, 599.33, 184.76, 334.18, 721.28, 356.85, 270.62, 475.92, 526.47, 165.8, 641.5, 361.85],
  [950.08, 785.41, 127.0, 73.44, 688.6, 228.04, 279.03, 876.39, 258.65, 870.94, 271.41, 355.27, 854.97, 286.67, 748.42, 760.9, 730.08, 857.64, 240.41, 443.42, 422.43, 543.66, 647.3, 595.19, 104.31, 480.28, 555.2, 409.12, 391.34, 196.33, 668.52, 161.38, 626.2, 841.96, 588.57, 203.06, 449.2, 424.13, 159.01, 753.49],
  [927.4, 949.96, 680.35, 629.78, 774.1, 801.79, 700.69, 839.53, 534.61, 726.3, 412.78, 858.45, 732.56, 300.47, 862.06, 635.57, 858.55, 854.77, 740.02, 841.62, 375.37, 98.95, 565.89, 368.08, 479.26, 933.34, 34.01, 367.76, 237.9, 618.66, 227.25, 416.84, 938.22, 412.34, 491.39, 635.69, 737.33, 387.24, 707.37, 545.7],
  [617.01, 337.15, 598.26, 476.62, 374.01, 479.43, 262.84, 583.93, 799.14, 683.52, 498.79, 267.15, 653.77, 639.52, 367.05, 612.47, 343.07, 540.51, 319.24, 100.96, 540.77, 818.7, 537.4, 661.3, 609.68, 156.52, 869.07, 543.52, 832.51, 728.11, 847.6, 591.12, 99.62, 930.39, 546.59, 342.96, 168.72, 530.41, 616.59, 686.33],
  [876.99, 879.17, 606.04, 547.88, 707.0, 720.74, 614.24, 788.89, 482.0, 688.43, 326.53, 772.29, 691.27, 216.39, 794.35, 590.87, 789.33, 800.05, 654.22, 755.97, 299.2, 70.21, 510.72, 322.62, 403.6, 847.03, 58.69, 290.01, 198.18, 555.9, 231.32, 336.43, 856.68, 426.64, 433.62, 549.44, 654.37, 311.23, 634.18, 511.1],
  [751.92, 605.18, 317.62, 185.87, 493.52, 338.35, 177.71, 677.49, 420.25, 675.34, 120.88, 334.69, 658.17, 233.75, 558.34, 566.42, 541.58, 659.54, 220.84, 334.9, 255.36, 461.63, 453.07, 431.23, 216.06, 413.17, 495.9, 245.73, 418.02, 388.27, 549.16, 177.12, 483.04, 698.79, 400.13, 111.95, 283.68, 253.21, 348.63, 569.2],
  [693.98, 693.27, 593.97, 497.32, 519.22, 667.19, 496.41, 606.09, 550.13, 519.74, 219.35, 665.9, 518.37, 205.35, 606.8, 415.64, 602.44, 614.52, 558.81, 614.59, 122.71, 162.12, 326.46, 159.33, 403.12, 716.03, 223.08, 119.01, 331.07, 590.12, 209.68, 320.4, 689.41, 372.63, 248.29, 448.63, 494.54, 133.93, 625.41, 352.12],
  [918.9, 803.5, 265.24, 217.51, 675.78, 384.69, 361.05, 838.06, 236.02, 803.32, 195.23, 481.96, 792.33, 141.06, 748.3, 692.01, 733.66, 828.07, 360.05, 528.24, 329.24, 392.48, 581.11, 486.4, 63.32, 589.73, 399.44, 313.83, 242.08, 249.73, 527.59, 31.83, 687.25, 710.26, 511.62, 280.35, 488.57, 334.86, 294.59, 663.45],
  [684.42, 426.05, 489.12, 369.54, 424.02, 372.63, 161.79, 640.04, 694.53, 717.85, 427.01, 169.17, 690.16, 561.03, 437.15, 634.04, 413.42, 601.2, 210.0, 8.49, 495.41, 761.69, 544.44, 636.91, 509.97, 110.94, 806.8, 495.15, 747.56, 620.29, 809.6, 499.14, 205.8, 913.19, 537.79, 243.19, 154.31, 486.43, 507.3, 692.74],
  [994.74, 981.4, 588.78, 558.62, 814.28, 725.24, 664.48, 906.71, 419.15, 809.35, 390.64, 809.9, 811.76, 253.36, 900.56, 710.86, 893.89, 916.02, 688.34, 818.32, 400.48, 184.81, 627.51, 443.21, 394.61, 900.06, 131.85, 388.9, 122.0, 511.2, 339.91, 346.97, 937.99, 531.27, 549.63, 591.69, 732.54, 412.39, 613.4, 632.26],
  [546.77, 676.5, 878.91, 767.51, 496.1, 927.32, 705.56, 465.28, 846.94, 317.72, 483.26, 875.21, 333.4, 502.76, 576.01, 259.93, 583.65, 496.58, 788.75, 772.8, 334.43, 349.36, 262.09, 158.15, 696.59, 884.9, 408.91, 345.78, 603.94, 886.48, 210.24, 613.37, 773.69, 150.84, 238.47, 683.05, 625.59, 335.81, 910.84, 147.09],
  [804.42, 525.95, 495.1, 404.12, 552.72, 343.46, 246.42, 766.83, 729.56, 853.25, 534.61, 127.09, 824.94, 657.08, 554.52, 771.76, 530.5, 725.33, 237.3, 141.69, 621.42, 876.51, 683.08, 770.53, 567.57, 33.06, 916.63, 619.47, 830.58, 638.58, 936.5, 575.54, 282.08, 1047.94, 675.43, 318.44, 292.83, 613.32, 505.33, 831.37],
  [490.94, 449.36, 613.86, 482.55, 279.31, 621.46, 374.52, 407.54, 668.24, 379.79, 237.41, 540.62, 364.77, 347.84, 365.51, 269.79, 359.56, 402.55, 464.61, 435.46, 135.42, 405.1, 156.21, 204.44, 474.68, 547.03, 467.27, 148.65, 533.71, 666.36, 396.51, 401.1, 463.86, 477.27, 109.88, 367.15, 291.5, 123.86, 645.2, 279.87],
  [575.14, 701.15, 875.64, 767.25, 519.95, 929.26, 713.34, 493.45, 835.62, 346.1, 482.37, 883.8, 361.84, 493.66, 600.83, 287.5, 607.91, 524.34, 794.55, 786.02, 335.74, 328.52, 284.43, 160.37, 690.26, 897.68, 386.46, 346.14, 586.79, 878.56, 184.62, 607.14, 792.8, 130.0, 254.95, 687.74, 640.12, 338.13, 907.48, 175.0],
  [821.72, 801.5, 537.61, 467.21, 634.2, 640.81, 522.95, 734.06, 447.15, 649.15, 234.8, 682.43, 648.09, 135.69, 720.37, 545.36, 713.71, 740.55, 565.42, 664.26, 220.29, 121.59, 454.36, 285.66, 335.84, 755.34, 147.87, 208.95, 202.43, 505.29, 259.7, 261.19, 768.07, 450.59, 375.89, 459.44, 564.66, 232.23, 567.21, 479.15],
  [1126.85, 1004.04, 262.64, 321.14, 882.61, 433.1, 523.76, 1045.68, 35.06, 1005.33, 402.84, 601.02, 995.9, 314.01, 953.3, 894.38, 937.82, 1036.07, 489.46, 690.74, 528.78, 513.14, 785.18, 671.36, 183.2, 729.37, 493.58, 513.02, 264.75, 136.52, 668.77, 239.79, 867.39, 862.18, 713.26, 443.99, 678.43, 535.77, 274.59, 857.07],
  [1006.43, 853.49, 128.41, 145.5, 749.18, 275.66, 353.19, 930.26, 183.93, 914.18, 303.73, 426.12, 900.12, 281.82, 812.47, 803.18, 794.91, 914.19, 313.85, 518.11, 451.27, 531.95, 690.05, 617.64, 77.67, 553.74, 534.54, 436.77, 347.35, 132.88, 668.47, 159.31, 699.91, 850.02, 626.47, 275.93, 519.48, 454.82, 155.63, 786.41],
  [1127.18, 1039.83, 388.91, 420.53, 900.61, 554.16, 600.17, 1042.4, 159.84, 980.94, 414.69, 701.79, 975.67, 289.46, 978.09, 872.45, 965.27, 1039.28, 583.82, 768.48, 511.24, 426.16, 769.66, 627.02, 254.03, 821.38, 393.56, 495.69, 158.91, 270.84, 586.91, 271.74, 930.99, 782.47, 693.27, 518.79, 731.78, 520.48, 404.43, 819.33],
  [994.68, 903.51, 335.18, 323.6, 764.66, 479.99, 478.2, 910.76, 196.86, 857.17, 278.36, 596.49, 850.0, 160.65, 841.72, 747.27, 828.81, 905.94, 475.15, 645.05, 382.48, 354.93, 641.47, 513.56, 150.72, 706.75, 342.61, 366.72, 138.77, 264.56, 508.05, 140.7, 800.25, 700.72, 566.61, 397.61, 598.82, 390.91, 359.24, 702.3],
  [920.74, 724.17, 151.86, 46.96, 652.08, 134.74, 198.7, 853.87, 366.3, 871.36, 316.23, 239.0, 851.75, 375.18, 700.87, 764.97, 680.4, 828.94, 132.19, 348.01, 461.1, 630.02, 652.87, 637.4, 220.08, 369.58, 649.94, 450.41, 504.59, 281.37, 739.69, 258.91, 540.28, 900.31, 605.03, 145.0, 385.92, 459.74, 174.9, 775.05],
  [284.78, 461.28, 938.16, 806.33, 303.96, 938.28, 675.01, 205.79, 974.7, 65.12, 553.0, 826.11, 71.84, 633.2, 363.27, 56.01, 378.11, 243.02, 771.29, 683.17, 414.45, 568.1, 168.27, 301.42, 791.01, 793.79, 633.9, 430.18, 785.76, 984.93, 459.85, 712.68, 620.44, 393.94, 230.84, 683.42, 533.69, 407.91, 969.52, 119.54],
  [638.19, 432.14, 442.63, 305.55, 367.78, 376.03, 99.54, 578.79, 607.76, 626.58, 287.17, 240.6, 602.03, 426.31, 409.22, 531.46, 388.17, 548.18, 196.56, 146.77, 351.55, 617.6, 430.76, 499.14, 409.31, 249.9, 663.87, 350.84, 619.18, 552.8, 666.32, 381.41, 279.16, 777.0, 411.17, 155.44, 93.35, 342.88, 467.61, 574.62],
  [971.67, 678.24, 577.12, 525.38, 732.15, 405.15, 420.52, 942.18, 829.23, 1038.71, 708.49, 257.93, 1009.65, 819.25, 722.8, 960.79, 699.04, 898.05, 378.89, 332.75, 807.07, 1051.87, 874.31, 960.55, 698.23, 220.05, 1087.32, 804.0, 977.32, 723.8, 1121.48, 723.09, 433.12, 1238.42, 867.32, 480.37, 483.89, 799.58, 577.01, 1022.86],
  [964.25, 682.26, 483.0, 434.7, 712.25, 311.06, 350.69, 926.75, 735.28, 1009.93, 633.4, 180.31, 982.15, 737.22, 714.28, 925.17, 690.26, 885.38, 296.08, 296.72, 741.51, 975.75, 832.05, 901.68, 607.91, 193.01, 1008.23, 737.11, 889.39, 629.67, 1053.61, 636.54, 436.17, 1179.85, 817.53, 402.19, 445.05, 734.89, 483.0, 978.69],
  [557.01, 279.87, 631.21, 503.53, 316.0, 523.16, 285.56, 524.45, 819.63, 627.31, 491.49, 317.72, 597.21, 635.04, 307.04, 559.77, 283.06, 480.71, 354.94, 142.64, 516.82, 798.54, 489.94, 625.43, 624.57, 215.7, 851.79, 521.17, 831.58, 755.38, 816.88, 598.2, 62.63, 889.76, 505.13, 361.77, 148.49, 505.87, 651.56, 637.97],
  [115.0, 198.13, 960.28, 820.15, 165.89, 906.02, 630.01, 108.9, 1069.17, 272.52, 643.09, 736.03, 242.47, 766.41, 135.0, 286.53, 159.01, 59.2, 725.93, 563.56, 552.86, 790.25, 323.88, 525.84, 866.79, 656.4, 856.72, 567.09, 952.71, 1046.37, 719.34, 803.59, 425.69, 687.98, 397.98, 672.81, 444.62, 541.91, 988.19, 399.8],
  [689.04, 834.05, 958.03, 860.88, 653.79, 1028.57, 829.56, 611.73, 885.24, 453.28, 578.07, 1001.12, 474.07, 559.91, 733.5, 412.64, 741.32, 646.74, 905.47, 912.84, 441.48, 342.86, 419.09, 277.06, 763.32, 1023.48, 384.48, 449.03, 611.02, 942.77, 182.32, 681.98, 926.74, 17.03, 387.84, 796.51, 769.33, 446.31, 989.26, 299.59],
  [985.47, 825.1, 113.53, 109.55, 725.6, 245.2, 319.63, 910.71, 219.02, 900.51, 294.45, 390.16, 885.41, 290.3, 786.76, 789.95, 768.7, 893.08, 277.97, 483.51, 444.42, 544.74, 676.47, 614.36, 91.26, 517.82, 551.37, 430.46, 373.75, 156.98, 676.12, 164.56, 666.87, 854.06, 615.36, 243.73, 489.43, 447.05, 144.07, 778.07],
  [830.96, 717.59, 306.49, 216.78, 587.44, 390.31, 303.27, 750.6, 323.56, 720.66, 108.91, 445.07, 708.5, 119.0, 660.55, 609.32, 646.24, 739.95, 323.99, 465.09, 252.0, 372.24, 497.32, 418.75, 130.17, 538.08, 393.87, 237.33, 287.93, 324.5, 488.43, 56.64, 612.11, 660.53, 430.26, 227.38, 409.7, 256.04, 338.14, 587.43],
  [847.07, 553.29, 582.84, 502.9, 610.87, 420.74, 352.08, 819.04, 824.62, 920.08, 639.59, 221.14, 890.52, 762.92, 598.59, 846.01, 574.91, 774.35, 339.01, 228.79, 721.62, 980.72, 764.39, 865.33, 669.63, 120.6, 1021.71, 720.44, 935.7, 728.7, 1036.5, 680.42, 308.16, 1141.29, 763.62, 424.12, 376.19, 713.06, 589.55, 913.57],
  [379.44, 524.17, 891.24, 764.91, 351.3, 908.02, 656.79, 297.24, 906.64, 158.35, 496.79, 817.16, 168.59, 560.84, 423.65, 94.64, 434.58, 329.19, 749.57, 688.84, 350.08, 475.11, 147.33, 213.93, 731.32, 801.44, 540.29, 365.25, 701.91, 925.79, 363.11, 650.49, 653.51, 307.83, 179.1, 653.76, 537.63, 345.6, 923.06, 22.85],
  [367.16, 370.39, 728.28, 593.08, 187.52, 719.18, 454.99, 282.86, 793.0, 267.16, 361.88, 608.61, 247.88, 470.64, 275.15, 164.2, 275.98, 280.51, 551.18, 474.8, 251.59, 492.74, 68.01, 242.86, 598.06, 587.29, 558.35, 266.44, 651.16, 788.36, 446.21, 525.68, 449.38, 475.69, 105.57, 464.81, 323.6, 241.22, 759.05, 217.17],
  [972.36, 877.01, 319.8, 299.64, 739.85, 459.23, 450.32, 888.96, 205.5, 839.0, 253.24, 570.32, 831.06, 143.56, 816.22, 728.66, 803.03, 883.18, 448.72, 617.02, 363.05, 355.98, 621.66, 500.13, 128.14, 679.4, 348.93, 347.24, 158.94, 261.08, 505.44, 112.54, 772.17, 696.32, 547.66, 369.85, 570.94, 371.01, 345.3, 686.97],
  [707.51, 769.07, 741.0, 649.95, 586.58, 820.66, 645.35, 620.6, 668.45, 497.94, 373.23, 816.12, 506.08, 340.72, 674.07, 412.7, 674.85, 640.95, 711.44, 752.69, 257.03, 145.0, 358.94, 157.0, 544.52, 858.07, 205.14, 259.52, 405.94, 723.4, 58.05, 463.82, 805.07, 230.0, 293.55, 601.24, 622.04, 265.42, 771.95, 316.84],
  [440.25, 330.13, 621.19, 481.92, 184.29, 591.22, 319.51, 368.88, 723.55, 399.06, 303.91, 466.9, 375.26, 438.96, 259.62, 304.2, 247.95, 347.81, 417.39, 332.28, 255.13, 535.16, 209.39, 332.79, 520.5, 445.03, 595.44, 264.83, 635.41, 700.4, 529.46, 459.8, 332.09, 595.86, 211.48, 341.66, 181.18, 243.1, 650.46, 357.88],
  [578.0, 572.68, 614.39, 497.72, 397.19, 656.14, 443.4, 490.72, 618.99, 421.28, 216.08, 614.96, 415.23, 275.43, 484.93, 312.05, 481.11, 495.94, 521.24, 535.65, 65.03, 280.14, 211.93, 113.84, 444.57, 643.48, 343.69, 78.77, 438.27, 638.76, 277.51, 362.92, 585.68, 388.85, 133.87, 413.73, 401.66, 64.78, 646.41, 272.59],
  [710.01, 427.73, 554.79, 446.29, 464.61, 418.42, 251.45, 676.11, 774.01, 771.29, 522.2, 198.2, 742.05, 655.82, 460.04, 695.85, 436.06, 633.14, 279.31, 91.79, 586.87, 855.97, 614.51, 722.14, 596.34, 67.12, 901.68, 587.35, 840.9, 692.57, 899.81, 590.57, 183.41, 996.18, 615.95, 332.07, 228.43, 577.5, 569.3, 763.74],
  [967.68, 740.1, 204.41, 167.74, 697.37, 50.33, 231.16, 908.3, 454.14, 945.17, 428.45, 172.79, 922.92, 497.12, 733.28, 843.13, 711.0, 877.94, 133.24, 330.77, 566.7, 750.94, 734.2, 741.76, 337.02, 313.65, 772.05, 557.47, 623.71, 351.32, 856.0, 381.1, 527.73, 1010.55, 694.86, 218.8, 414.48, 563.87, 209.88, 865.45],
  [813.46, 612.13, 263.08, 131.61, 543.74, 208.78, 87.46, 749.14, 454.8, 777.03, 275.96, 186.95, 755.65, 374.82, 589.89, 673.67, 569.14, 722.17, 66.84, 247.29, 403.13, 614.73, 564.0, 575.96, 276.0, 291.76, 645.52, 395.34, 537.97, 383.56, 703.8, 282.47, 433.5, 848.61, 524.3, 48.76, 273.93, 399.17, 286.92, 694.86],
  [349.21, 369.09, 750.75, 615.62, 185.53, 741.51, 476.66, 263.91, 813.66, 244.66, 382.77, 629.14, 225.33, 489.04, 271.77, 142.59, 274.57, 264.15, 573.07, 492.93, 268.64, 501.98, 55.8, 246.13, 619.45, 605.15, 568.03, 283.77, 667.31, 810.12, 448.37, 546.51, 460.48, 468.39, 110.55, 487.23, 341.87, 258.64, 781.54, 202.25],
  [476.11, 438.4, 626.98, 495.01, 266.63, 632.16, 382.31, 392.67, 683.03, 366.23, 252.12, 547.09, 350.78, 362.49, 353.27, 256.6, 347.91, 387.83, 473.51, 437.83, 148.77, 415.0, 143.18, 206.07, 489.2, 549.87, 477.76, 162.35, 547.73, 680.66, 401.31, 415.84, 459.76, 476.0, 102.18, 377.48, 292.08, 137.38, 658.24, 270.66],
  [1022.76, 893.07, 207.19, 223.52, 774.55, 361.47, 415.54, 943.04, 133.21, 911.47, 301.5, 505.29, 900.19, 238.84, 843.77, 800.13, 827.89, 931.4, 389.02, 583.45, 437.54, 472.97, 688.93, 592.36, 71.87, 628.23, 467.77, 422.09, 266.41, 147.0, 618.53, 138.31, 757.17, 806.36, 619.89, 334.96, 567.07, 443.2, 230.08, 771.37],
  [978.15, 909.44, 406.06, 383.47, 760.66, 545.67, 517.91, 892.46, 259.09, 826.52, 283.13, 648.12, 821.82, 143.59, 841.27, 718.52, 830.17, 891.61, 525.97, 681.31, 360.44, 289.02, 617.35, 472.14, 214.27, 750.96, 271.88, 345.19, 74.85, 335.84, 445.89, 181.04, 824.99, 640.4, 540.28, 439.5, 620.31, 370.3, 430.84, 664.3],
  [712.9, 731.3, 635.93, 544.54, 553.94, 715.67, 549.48, 624.8, 574.82, 525.59, 270.44, 718.78, 527.44, 238.17, 642.0, 426.57, 639.13, 637.13, 610.97, 667.11, 172.72, 122.05, 347.64, 159.15, 440.84, 769.01, 187.54, 170.67, 333.38, 623.08, 158.9, 359.39, 737.82, 331.85, 271.67, 500.85, 545.15, 183.31, 667.01, 350.77],
  [703.44, 829.33, 909.81, 815.69, 647.03, 984.75, 793.43, 623.03, 832.5, 470.94, 534.69, 965.14, 488.98, 509.83, 729.4, 418.23, 735.56, 654.88, 866.45, 883.89, 403.04, 289.29, 410.01, 247.81, 713.68, 993.36, 331.28, 409.37, 557.43, 891.46, 129.0, 632.91, 908.17, 68.15, 369.63, 756.86, 743.27, 408.83, 940.86, 305.37],
  [699.98, 427.31, 522.35, 409.48, 447.15, 393.32, 210.04, 661.34, 736.58, 748.92, 479.89, 176.95, 720.34, 613.37, 450.36, 669.62, 426.36, 620.1, 244.39, 50.25, 546.84, 814.41, 584.44, 685.2, 556.34, 73.16, 859.7, 546.94, 798.75, 657.79, 860.49, 548.93, 190.71, 960.42, 582.23, 291.03, 194.98, 537.66, 538.31, 733.43],
  [719.12, 660.17, 466.89, 364.93, 503.29, 534.29, 375.41, 634.14, 457.06, 580.67, 88.62, 541.33, 572.41, 118.83, 586.05, 470.11, 576.62, 631.92, 430.19, 509.1, 104.4, 257.4, 363.27, 258.52, 285.57, 603.57, 300.85, 88.6, 306.47, 478.66, 341.48, 202.79, 611.79, 503.09, 289.03, 320.54, 407.71, 112.7, 498.69, 434.36],
  [812.43, 594.6, 290.42, 171.29, 542.03, 201.8, 75.19, 752.35, 498.83, 791.53, 322.52, 134.46, 768.63, 426.86, 580.94, 691.29, 559.18, 722.36, 24.08, 207.97, 442.06, 663.78, 584.32, 611.77, 326.5, 241.02, 696.21, 435.46, 591.07, 420.58, 747.82, 335.56, 400.94, 886.97, 550.58, 90.52, 261.66, 437.12, 310.5, 720.2],
  [853.71, 700.21, 221.94, 106.98, 594.61, 278.86, 221.82, 779.07, 326.96, 772.65, 180.11, 341.53, 756.56, 231.93, 657.61, 662.77, 640.18, 761.33, 219.34, 390.16, 330.39, 485.21, 549.17, 505.48, 128.16, 447.64, 507.01, 317.98, 381.41, 286.93, 595.97, 126.09, 559.09, 760.98, 491.58, 140.43, 369.68, 331.02, 253.71, 657.84],
  [794.57, 654.29, 286.88, 166.21, 539.18, 332.02, 214.77, 718.47, 371.62, 708.36, 119.1, 359.81, 692.65, 201.56, 605.83, 598.26, 589.48, 702.37, 240.42, 377.7, 267.37, 443.52, 484.66, 443.22, 167.2, 449.59, 472.12, 255.55, 376.14, 344.76, 543.74, 129.12, 531.37, 702.98, 426.64, 139.46, 333.11, 267.42, 318.64, 593.04],
  [576.97, 275.88, 717.04, 596.04, 370.84, 593.03, 381.47, 560.54, 917.93, 684.12, 600.8, 375.53, 652.8, 744.17, 335.93, 629.87, 313.83, 512.61, 437.84, 219.8, 623.17, 905.83, 573.1, 723.14, 726.68, 246.01, 959.93, 628.16, 940.26, 847.54, 918.34, 704.28, 58.31, 980.82, 597.44, 460.84, 257.75, 612.0, 734.77, 717.72],
  [622.49, 753.01, 897.66, 794.31, 571.76, 959.25, 751.65, 542.01, 843.24, 390.62, 509.66, 922.84, 408.17, 507.12, 652.69, 337.72, 659.76, 574.13, 830.16, 830.41, 367.49, 317.97, 335.95, 197.06, 707.64, 941.48, 370.55, 376.43, 582.99, 892.5, 162.2, 625.04, 842.34, 82.13, 303.46, 722.11, 685.99, 371.22, 929.27, 224.64],
  [1122.48, 1023.65, 346.16, 383.99, 889.77, 512.77, 570.64, 1038.74, 117.48, 984.16, 403.21, 665.8, 977.49, 288.14, 965.26, 874.6, 951.57, 1033.5, 549.26, 739.07, 510.36, 449.79, 769.38, 636.03, 222.41, 788.01, 422.17, 494.62, 187.68, 226.98, 609.29, 251.38, 906.09, 804.58, 694.21, 489.33, 709.47, 518.91, 361.04, 826.71],
  [783.06, 923.43, 983.41, 896.43, 741.94, 1067.48, 885.62, 705.7, 886.54, 547.15, 619.72, 1057.2, 568.11, 580.0, 823.14, 505.54, 830.09, 740.37, 955.83, 980.15, 494.61, 338.65, 505.54, 345.06, 783.88, 1089.06, 365.31, 499.75, 599.69, 954.17, 194.2, 705.36, 1006.05, 107.3, 467.33, 845.81, 840.5, 501.17, 1013.86, 392.3],
  [550.78, 438.46, 518.15, 381.5, 300.13, 509.19, 256.32, 475.08, 608.13, 479.24, 189.62, 422.69, 459.91, 328.89, 374.29, 373.55, 361.22, 458.63, 347.54, 327.56, 177.34, 459.23, 263.01, 310.23, 405.69, 436.7, 514.04, 181.6, 527.64, 588.57, 485.16, 343.92, 390.81, 587.82, 227.88, 254.16, 195.43, 166.63, 548.5, 396.74],
  [355.05, 115.38, 762.95, 624.87, 130.6, 685.96, 416.75, 324.32, 909.92, 442.47, 514.51, 501.81, 411.09, 655.67, 105.08, 393.86, 81.06, 279.18, 506.97, 326.5, 478.88, 756.48, 353.95, 531.86, 705.87, 416.39, 818.0, 488.81, 854.38, 867.42, 734.89, 657.71, 193.35, 766.83, 396.23, 474.12, 227.61, 466.84, 788.1, 489.8],
  [360.85, 223.78, 704.16, 564.03, 90.8, 653.65, 376.95, 302.17, 823.8, 375.56, 409.98, 499.4, 346.83, 546.24, 153.94, 303.21, 140.87, 270.42, 474.32, 339.55, 358.1, 633.06, 241.13, 408.9, 619.69, 446.52, 695.18, 368.91, 742.7, 793.78, 611.41, 562.88, 275.83, 652.1, 274.7, 416.6, 199.56, 346.07, 731.96, 385.43],
  [747.44, 665.15, 410.03, 307.75, 518.28, 477.87, 333.22, 664.34, 414.57, 622.43, 46.32, 494.77, 612.09, 102.36, 597.53, 511.19, 586.0, 658.22, 380.43, 476.88, 148.33, 301.42, 401.12, 313.36, 233.32, 565.76, 337.52, 133.15, 301.86, 427.6, 396.77, 152.16, 595.12, 560.55, 330.51, 272.15, 389.19, 153.63, 441.89, 483.52],
  [871.58, 920.34, 742.4, 678.46, 740.08, 851.98, 723.83, 784.56, 614.97, 660.57, 435.11, 888.07, 669.66, 346.21, 828.15, 576.84, 827.04, 804.05, 773.41, 853.32, 365.57, 92.02, 519.2, 316.81, 539.79, 951.14, 87.21, 361.57, 322.01, 692.53, 145.58, 469.92, 931.97, 323.07, 449.44, 665.56, 737.22, 376.63, 770.86, 479.5],
  [613.09, 456.58, 447.35, 307.69, 348.86, 421.81, 163.88, 543.35, 569.22, 563.99, 194.02, 331.4, 542.78, 338.21, 410.13, 461.09, 392.91, 520.77, 255.88, 253.02, 244.57, 512.96, 352.75, 396.36, 364.9, 356.64, 561.36, 243.89, 536.34, 533.91, 559.46, 318.31, 356.01, 674.58, 320.32, 168.47, 150.01, 236.03, 476.29, 488.97],
  [951.03, 667.24, 496.33, 443.68, 700.81, 325.08, 350.22, 914.76, 747.77, 1000.33, 635.18, 181.95, 972.29, 741.74, 701.03, 916.85, 677.02, 872.92, 300.51, 287.17, 740.31, 978.08, 825.05, 898.72, 616.68, 180.71, 1011.66, 736.3, 896.75, 643.2, 1053.27, 643.11, 420.86, 1176.94, 812.23, 404.85, 436.72, 733.43, 497.2, 972.21],
  [506.45, 573.65, 752.64, 634.46, 390.08, 789.51, 561.49, 418.81, 748.91, 312.71, 354.32, 730.75, 314.59, 401.8, 476.03, 214.85, 479.11, 437.07, 646.08, 629.69, 203.3, 324.24, 156.08, 54.08, 581.11, 741.42, 390.88, 216.92, 540.07, 774.73, 245.13, 498.74, 642.07, 284.27, 103.35, 541.76, 484.01, 202.0, 784.65, 142.89],
  [1057.9, 1001.62, 468.3, 465.95, 848.91, 620.25, 612.25, 971.2, 273.16, 894.69, 377.4, 737.43, 892.46, 234.64, 930.99, 789.31, 920.71, 973.23, 615.65, 776.61, 441.32, 309.31, 693.12, 532.27, 292.96, 844.08, 271.77, 426.76, 44.2, 372.77, 470.39, 273.56, 921.12, 665.52, 614.96, 533.04, 716.3, 451.98, 489.35, 725.76],
  [202.11, 102.49, 930.43, 791.04, 162.31, 860.04, 588.43, 207.45, 1061.05, 366.43, 645.96, 676.11, 335.39, 778.66, 84.22, 363.98, 102.62, 157.04, 680.54, 500.0, 576.56, 833.01, 377.06, 578.28, 856.77, 584.07, 898.36, 589.43, 971.89, 1027.66, 778.26, 800.44, 343.01, 765.65, 443.55, 640.82, 399.08, 564.91, 956.67, 475.96],
  [650.82, 738.18, 796.68, 697.5, 554.62, 865.06, 670.87, 565.41, 738.49, 432.79, 414.56, 842.58, 443.71, 403.0, 640.34, 355.95, 643.65, 589.93, 744.09, 763.93, 280.48, 221.69, 319.39, 132.48, 604.53, 872.63, 280.24, 287.02, 481.79, 788.18, 87.93, 522.3, 797.33, 170.26, 265.32, 634.64, 625.64, 286.25, 828.11, 252.46],
  [945.18, 765.1, 115.39, 28.3, 679.69, 176.87, 246.59, 874.49, 301.68, 879.45, 295.96, 305.08, 861.8, 330.95, 734.54, 770.76, 715.14, 852.84, 194.55, 405.23, 446.01, 588.1, 657.43, 621.31, 158.15, 433.89, 602.89, 433.76, 444.92, 223.71, 707.13, 208.34, 593.38, 875.61, 603.31, 178.19, 426.55, 446.36, 145.25, 771.11],
  [605.14, 718.85, 852.31, 747.36, 536.32, 911.67, 703.41, 522.12, 804.68, 379.01, 462.52, 874.64, 393.3, 465.17, 619.12, 314.13, 624.95, 551.04, 781.88, 783.54, 319.42, 291.78, 299.21, 148.95, 664.05, 894.34, 349.16, 328.57, 551.77, 850.59, 147.41, 581.17, 799.64, 129.4, 260.43, 673.89, 639.94, 323.01, 884.01, 203.67],
  [660.13, 419.02, 470.03, 342.48, 394.14, 370.84, 125.54, 610.22, 661.88, 678.18, 374.16, 189.86, 651.44, 510.3, 417.43, 590.44, 394.35, 574.03, 196.08, 57.25, 440.2, 707.05, 497.11, 582.98, 471.39, 163.78, 752.78, 439.94, 699.49, 594.43, 754.48, 454.44, 222.64, 859.8, 486.71, 204.8, 111.76, 431.25, 490.92, 644.46],
  [623.17, 483.13, 440.4, 302.65, 364.16, 430.0, 186.94, 550.11, 546.21, 559.13, 157.16, 357.75, 539.62, 301.8, 430.73, 453.34, 414.85, 530.74, 272.62, 289.87, 210.11, 475.14, 342.29, 368.05, 341.88, 391.53, 523.14, 208.34, 500.61, 517.73, 525.16, 289.55, 393.08, 645.86, 302.33, 175.03, 187.17, 202.25, 470.37, 472.63],
  [1051.79, 1004.78, 497.41, 490.15, 848.65, 647.09, 629.16, 964.65, 304.93, 883.24, 384.55, 758.63, 882.08, 240.07, 931.82, 779.22, 922.29, 968.31, 636.58, 791.95, 437.67, 286.34, 685.61, 518.83, 317.91, 862.36, 244.98, 423.6, 40.45, 404.19, 446.92, 292.26, 931.81, 641.41, 607.2, 550.89, 726.32, 448.71, 519.05, 712.04],
  [622.09, 432.26, 447.0, 307.68, 352.23, 394.37, 119.6, 559.2, 597.74, 599.0, 256.3, 273.85, 575.35, 397.85, 400.71, 501.5, 380.83, 531.02, 217.81, 182.85, 314.43, 582.44, 398.53, 461.86, 396.25, 287.31, 629.8, 314.01, 593.12, 549.49, 629.05, 361.64, 299.68, 739.81, 375.93, 157.41, 99.4, 305.64, 473.52, 540.83],
  [914.92, 705.07, 183.37, 95.63, 644.85, 109.29, 180.42, 851.32, 411.89, 878.3, 346.71, 192.44, 857.31, 417.23, 688.27, 774.03, 666.99, 823.86, 96.61, 314.8, 486.49, 669.84, 663.43, 662.1, 268.89, 326.46, 692.32, 476.87, 552.53, 321.71, 774.02, 304.4, 510.27, 929.47, 620.36, 146.58, 369.84, 484.06, 200.88, 790.98],
  [853.19, 729.05, 272.01, 185.07, 605.2, 358.67, 291.76, 774.04, 302.65, 749.13, 137.06, 424.82, 736.16, 147.63, 675.87, 637.89, 660.66, 761.69, 302.85, 457.11, 283.99, 403.55, 525.29, 452.4, 102.4, 524.25, 422.46, 269.61, 300.51, 294.92, 522.36, 48.17, 612.32, 695.04, 459.91, 212.79, 413.01, 287.45, 303.69, 619.04],
  [788.5, 916.54, 948.52, 863.69, 733.98, 1035.34, 858.82, 709.09, 848.76, 554.35, 589.11, 1030.14, 573.64, 544.81, 816.75, 505.38, 822.64, 741.76, 927.0, 957.75, 468.14, 301.12, 496.79, 326.17, 748.39, 1065.71, 326.99, 472.43, 561.43, 917.29, 161.01, 670.47, 990.23, 121.08, 453.48, 816.84, 820.24, 475.27, 978.8, 392.31],
  [652.38, 398.48, 502.93, 378.35, 391.0, 394.85, 163.51, 607.12, 699.87, 684.53, 412.23, 198.66, 656.81, 549.45, 405.88, 601.16, 382.27, 568.63, 225.61, 29.15, 472.33, 742.54, 512.52, 609.55, 510.7, 141.82, 789.39, 472.87, 739.28, 630.18, 785.4, 494.45, 188.44, 884.94, 507.68, 243.76, 122.1, 462.96, 522.64, 661.08],
  [836.75, 630.58, 247.24, 123.02, 566.72, 183.1, 104.48, 773.26, 450.25, 802.7, 297.84, 174.63, 781.17, 390.94, 611.06, 699.56, 590.0, 745.68, 52.5, 254.75, 427.67, 634.28, 590.01, 601.04, 279.21, 289.04, 663.36, 419.57, 548.24, 373.53, 726.38, 293.03, 445.22, 873.04, 550.34, 74.53, 293.53, 423.95, 269.42, 720.91],
  [347.14, 519.55, 936.26, 808.27, 355.07, 948.1, 692.26, 269.56, 956.02, 115.8, 543.6, 849.4, 132.31, 610.59, 420.27, 91.29, 433.63, 306.86, 786.5, 714.74, 398.3, 522.83, 178.54, 263.9, 779.19, 826.71, 587.52, 413.62, 751.96, 973.67, 405.23, 698.78, 666.3, 330.19, 222.77, 693.26, 563.81, 393.39, 967.98, 70.71],
  [665.3, 622.48, 518.25, 409.02, 458.8, 574.3, 391.06, 579.53, 516.09, 522.02, 124.19, 561.14, 514.22, 175.41, 543.64, 411.67, 535.88, 579.38, 456.27, 510.06, 47.51, 257.16, 306.02, 204.01, 342.44, 610.53, 310.26, 32.45, 353.26, 536.25, 312.68, 260.25, 594.45, 460.24, 230.8, 346.13, 394.99, 57.69, 550.2, 375.7],
  [523.36, 450.02, 564.69, 431.67, 292.07, 568.23, 321.6, 442.83, 631.35, 428.18, 200.86, 488.79, 411.43, 325.1, 374.18, 319.48, 364.93, 432.81, 411.1, 390.45, 135.13, 417.63, 206.49, 246.17, 433.78, 500.79, 476.44, 144.17, 518.31, 623.08, 428.08, 363.9, 436.16, 522.87, 163.37, 313.7, 251.68, 123.2, 595.79, 333.47],
  [841.4, 675.01, 221.79, 89.05, 578.33, 251.25, 182.48, 769.22, 358.37, 772.0, 199.55, 299.34, 754.4, 269.54, 637.38, 663.38, 619.01, 748.97, 177.16, 350.87, 345.73, 519.46, 550.11, 521.99, 165.89, 405.62, 544.05, 334.57, 423.76, 307.04, 624.01, 168.31, 524.25, 783.61, 497.04, 101.41, 340.31, 344.95, 252.36, 665.68],
]

# ������ (I)
wi = [
    3, 5, 7, 6, 3, 9, 1, 5, 3, 6, 9, 2, 5, 7, 2, 1, 9, 5, 8, 7, 2, 4, 4, 7, 1, 2, 4, 5, 6, 9, 
    8, 5, 7, 3, 5, 6, 8, 3, 4, 5, 6, 8, 9, 9, 1, 2, 4, 5, 6, 7, 8, 9, 9, 4, 1, 3, 3, 3, 4, 5, 
    9, 6, 7, 7, 1, 3, 4, 8, 5, 3, 5, 6, 7, 8, 8, 3, 5, 2, 5, 3
]

# ������ҽԺ��λ�����꣨���û��ṩ�������ж�ȡ��
demand_points = [(249, 637), (377, 167), (22, 621), (884, 356), (97, 578), (468, 344), (279, 626), 
 (272, 284), (794, 294), (6, 498), (336, 918), (854, 169), (523, 636), (308, 913), 
 (181, 541), (130, 132), (305, 147), (35, 227), (155, 292), (492, 138), (596, 952), 
 (671, 370), (952, 4), (858, 9), (883, 416), (879, 858), (191, 980), (341, 148), 
 (337, 344), (952, 129), (504, 922), (603, 732), (183, 295), (205, 761), (651, 597), 
 (398, 654), (884, 263), (580, 53), (565, 223), (608, 754), (533, 647), (228, 185), 
 (121, 356), (239, 661), (178, 928), (843, 274), (334, 505), (615, 205), (412, 259), 
 (423, 323), (992, 407), (259, 930), (68, 198), (97, 983), (561, 524), (874, 618), 
 (751, 636), (347, 449), (50, 707), (586, 435), (872, 22), (407, 792), (25, 350), 
 (940, 780), (246, 826), (426, 144), (281, 887), (749, 326), (549, 445), (12, 379), 
 (645, 397), (539, 124), (343, 310), (92, 945), (789, 327), (567, 197), (533, 963), 
 (360, 558), (531, 583), (453, 248)]

facility_points = [(880, 973), (970, 682), (376, 40), (450, 159), (790, 718), (547, 15), (633, 278), 
 (801, 934), (128, 97), (644, 996), (392, 438), (731, 137), (665, 972), (248, 423), 
 (878, 723), (595, 896), (877, 699), (832, 894), (617, 181), (800, 300), (396, 589), 
 (118, 645), (563, 787), (362, 762), (289, 223), (856, 202), (56, 620), (387, 576), 
 (52, 385), (229, 38), (162, 800), (295, 306), (942, 437), (204, 991), (498, 743), 
 (552, 270), (736, 437), (408, 590), (375, 8), (483, 913)]


# ���ӻ�������ҽԺ��λ��
def plot_points():
    plt.figure(figsize=(8, 6))
    
    # ���������
    for i, (x, y) in enumerate(demand_points):
        plt.scatter(x, y, c='blue', marker='o')
        plt.text(x + 0.1, y + 0.1, f'Demand {i + 1}', fontsize=9)
    
    # ����ҽԺλ��
    for j, (x, y) in enumerate(facility_points):
        plt.scatter(x, y, c='red', marker='^')
        plt.text(x + 0.1, y + 0.1, f'Facility {j + 1}', fontsize=9)
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Locations of Demand Points and Facilities')
    plt.grid(True)
    plt.show()

# Ŀ�꺯���������ܸ���������
def objective_function(x):
    total_demand_covered = 0 # ��ʼ������Ϊ 0
    for i in range(I):
        is_covered = any(x[j] == 1 and dij[i][j] / V <= T for j in range(J)) # ����Ƿ񱻸���
        if is_covered:
            total_demand_covered += wi[i] # �������������ӵ��ܸ�����
    return total_demand_covered # �����ܸ���

# ��ʼ����Ⱥ
def initialize_population(size, bounds): # ����һ����ʼ����Ⱥ�ĺ����������������Ⱥ��ģ (size) �ͱ�����Χ (bounds)
    population = [] # ���б����ڴ洢���ɵĸ���
    for _ in range(size):
        individual = [random.choice([0, 1]) for _ in range(J)]
        # ����һ�����壬�ø����� J �����ѡ��� 0 �� 1 ��ɡ�
        if sum(individual) * C <= M:  # ����Ƿ�����Ԥ��Լ��
            population.append(individual)  # ������Ԥ��ĸ�����뵽 population ��
    return population

# ��Ӧ������
def evaluate_population(population):
    # ����Ⱥ�е�ÿ������ individual ���� objective_function���õ�����Ӧ��ֵ��
    # ������Щ��Ӧ��ֵ���һ���б�����
    return [objective_function(individual) for individual in population]

# ���̶�ѡ����
def select(population, fitness):
    min_fitness = min(fitness)
    if min_fitness < 0:
        fitness = [f - min_fitness for f in fitness]  # ��������Ӧ��ֵƽ��Ϊ�Ǹ�ֵ
    total_fitness = sum(fitness)
    
    # �������Ӧ��Ϊ 0�����������Ĵ���
    if total_fitness == 0:
        # ֱ�ӷ�����Ⱥ���������ѡ��һЩ����
        return random.choices(population, k=len(population))
    
    # ����ÿ�������ѡ�����
    probabilities = [f / total_fitness for f in fitness]
    selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
    return [population[i] for i in selected_indices]


# ����
def crossover(dad, mom):
    child1 = [(p1 + p2) // 2 for p1, p2 in zip(dad, mom)] # �Ӵ�1�Ļ���Ϊ������ĸ������ľ�ֵ
    child2 = [(p1 + p2) // 2 for p1, p2 in zip(dad, mom)] # �Ӵ�2�Ļ���Ϊ������ĸ������ľ�ֵ
    return child1, child2

# ����
def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        mutate_point = random.randint(0, J - 1) # ���ѡ��һ������λ�ý��б���
        individual[mutate_point] = 1 - individual[mutate_point]  # ��0��1֮���л�
    return individual

# �Ŵ��㷨������
def genetic_algorithm(bounds, population_size, generations, mutation_rate):
    population = initialize_population(population_size, bounds)
    best_fitness_over_time = []
    best_solution = None
    best_solution_value = -float('inf')

    for generation in range(generations):
        fitness = evaluate_population(population) # ������Ⱥ��Ӧ��
        best_fitness = max(fitness)
        best_fitness_over_time.append(best_fitness) # ÿһ�������Ӧ��
        
        # ������ѽ�
        if best_fitness > best_solution_value:
            best_solution_value = best_fitness
            best_solution = population[fitness.index(best_fitness)]
        
        # ��ӡÿ���������Ӧ��
        print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

        selected_population = select(population, fitness)
        new_population = [] # ���ڴ洢�Ӵ��õ��µ��б�������
        
        # ��������Ⱥ �������
        for i in range(0, len(selected_population), 2):
            dad, mom = selected_population[i], selected_population[i+1]
            child1, child2 = crossover(dad, mom)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))
        
        population = new_population

    # ������ѽ⼰����Ӧ�������ʱ��仯
    return best_solution, best_fitness_over_time

# ��������
bounds = [0, 1]  
population_size = 20
generations = 50
mutation_rate = 0.1

# ִ���Ŵ��㷨
best_solution, fitness_over_time = genetic_algorithm(bounds, population_size, generations, mutation_rate)

# �����ѽ�
best_solution_value = objective_function(best_solution)
print("Best Solution:", best_solution)
print("Maximum Coverage:", best_solution_value)
print("Number of Demand Points:", I)
print("Demand Value of Demand Points:", wi)
print("Total Demand Value:", sum(wi))

# ������Ӧ�ȱ仯ͼ
plt.plot(fitness_over_time)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Genetic Algorithm Optimization for Facility Location')
plt.show()

# ������������ʩ���λ��
plot_points()