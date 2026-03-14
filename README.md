```
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  HdRAG Behavioral Analysis
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

  GGUF:     gpt-oss-20b-heretic-v2.Q4_K_M
  HDC dims: 16384
  N-gram:   5
  Loading tokenizer... vocab=201,088

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Running: ngram  [ok]  (1.0s)  rho=+1.0000  rho=+1.0000  rho_med=+1.0000  rho_doc=+0.7882  mono=yes  1t=-0.0976  2t=-0.0369  3t=+0.0081  4t=+0.0616  5t=+0.1109

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  N-GRAM RETRIEVAL ANALYSIS
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

  PROMPT:      "What is a geometric shape that uses complex numbers to describe its convex curvatures"
  Tokens:      16
  HDC dims:    16384
  N-gram:      5
  Documents:   70

  TOKEN MAP
  ────────────────────────────────────────────────────────────
    [ 0] id=  4827  'What'
    [ 1] id=   382  ' is'
    [ 2] id=   261  ' a'
    [ 3] id= 82570  ' geometric'
    [ 4] id=  9591  ' shape'
    [ 5] id=   484  ' that'
    [ 6] id=  8844  ' uses'
    [ 7] id=  8012  ' complex'
    [ 8] id=  8663  ' numbers'
    [ 9] id=   316  ' to'
    [10] id= 12886  ' describe'
    [11] id=  1617  ' its'
    [12] id=142423  ' convex'
    [13] id=  4396  ' cur'
    [14] id=    85  'v'
    [15] id=  3351  'atures'
    1-token spans:  16
    2-token spans:  15
    3-token spans:  14
    4-token spans:  13
    5-token spans:  12

  QUERY VECTOR
  ──────────────────────────────────────────────────
  Active dims:     5756  (35.1%)
  Positive bits:   2922  (17.8%)
  Negative bits:   2834  (17.3%)
  Zero dims:      10628  (64.9%)

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1-TOKEN    avg=-0.0976  max=+0.0449  min=-0.1583  std=0.0644  n=16
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      label     score   agree   disag   engag   d_lit  rt  bar                                       span
       t1.1   +0.0442     611     474    1085    3250  ok               ████                         'What'
       t1.2   -0.1456     287     782    1069    3089  ok   ░░░░░░░░░░░░                             ' is'
       t1.3   -0.1553     293     807    1100    3103  ok  ░░░░░░░░░░░░░                             ' a'
       t1.4   -0.1583     306     851    1157    3294  ok  ░░░░░░░░░░░░░                             ' geometric'
       t1.5   -0.1478     287     769    1056    3085  ok   ░░░░░░░░░░░░                             ' shape'
       t1.6   -0.1163     326     718    1044    3101  ok     ░░░░░░░░░░                             ' that'
       t1.7   -0.1091     322     684    1006    3071  ok      ░░░░░░░░░                             ' uses'
       t1.8   -0.1196     359     754    1113    3109  ok     ░░░░░░░░░░                             ' complex'
       t1.9   -0.1060     367     718    1085    3086  ok      ░░░░░░░░░                             ' numbers'
      t1.10   -0.1320     311     747    1058    3108  ok    ░░░░░░░░░░░                             ' to'
      t1.11   -0.1241     324     731    1055    3102  ok     ░░░░░░░░░░                             ' describe'
      t1.12   -0.1344     306     744    1050    3037  ok    ░░░░░░░░░░░                             ' its'
      t1.13   -0.1193     374     772    1146    3382  ok     ░░░░░░░░░░                             ' convex'
      t1.14   -0.0813     392     659    1051    3103  ok        ░░░░░░░                             ' cur'
      t1.15   -0.0023     528     543    1071    3133  ok                                            'v'
      t1.16   +0.0449     562     427     989    3102  ok               ████                         'atures'

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  2-TOKEN    avg=-0.0369  max=+0.1747  min=-0.1008  std=0.0710  n=15
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      label     score   agree   disag   engag   d_lit  rt  bar                                       span
       t2.1   +0.1747     895     334    1229    3323  ok               ███████████████              'What is'
       t2.2   -0.0216     533     604    1137    3414  ok             ░░                             ' is a'
       t2.3   -0.0826     412     695    1107    3308  ok        ░░░░░░░                             ' a geometric'
       t2.4   -0.1008     435     770    1205    3546  ok       ░░░░░░░░                             ' geometric shape'
       t2.5   -0.0794     439     689    1128    3338  ok        ░░░░░░░                             ' shape that'
       t2.6   -0.0586     475     672    1147    3399  ok          ░░░░░                             ' that uses'
       t2.7   -0.0651     433     655    1088    3320  ok          ░░░░░                             ' uses complex'
       t2.8   -0.0632     479     684    1163    3336  ok          ░░░░░                             ' complex numbers'
       t2.9   -0.0508     489     650    1139    3322  ok           ░░░░                             ' numbers to'
      t2.10   -0.0681     443     669    1112    3359  ok         ░░░░░░                             ' to describe'
      t2.11   -0.0615     441     653    1094    3312  ok          ░░░░░                             ' describe its'
      t2.12   -0.0793     419     668    1087    3288  ok        ░░░░░░░                             ' its convex'
      t2.13   -0.0668     481     696    1177    3584  ok         ░░░░░░                             ' convex cur'
      t2.14   -0.0204     526     571    1097    3277  ok             ░░                             ' curv'
      t2.15   +0.0895     743     423    1166    3321  ok               ████████                     'vatures'

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  3-TOKEN    avg=+0.0081  max=+0.2355  min=-0.0510  std=0.0682  n=14
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      label     score   agree   disag   engag   d_lit  rt  bar                                       span
       t3.1   +0.2355    1164     298    1462    3572  ok               ████████████████████         'What is a'
       t3.2   +0.0309     687     549    1236    3537  ok               ███                          ' is a geometric'
       t3.3   -0.0215     615     651    1266    3603  ok             ░░                             ' a geometric shape'
       t3.4   -0.0510     592     714    1306    3741  ok           ░░░░                             ' geometric shape that'
       t3.5   -0.0064     645     605    1250    3627  ok              ░                             ' shape that uses'
       t3.6   -0.0175     632     654    1286    3603  ok              ░                             ' that uses complex'
       t3.7   -0.0200     597     632    1229    3593  ok             ░░                             ' uses complex numbers'
       t3.8   -0.0218     639     661    1300    3572  ok             ░░                             ' complex numbers to'
       t3.9   -0.0152     614     602    1216    3499  ok              ░                             ' numbers to describe'
      t3.10   -0.0202     601     632    1233    3576  ok             ░░                             ' to describe its'
      t3.11   -0.0130     626     624    1250    3580  ok              ░                             ' describe its convex'
      t3.12   -0.0092     609     575    1184    3501  ok              ░                             ' its convex cur'
      t3.13   -0.0210     638     656    1294    3800  ok             ░░                             ' convex curv'
      t3.14   +0.0632     766     487    1253    3535  ok               █████                        ' curvatures'

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  4-TOKEN    avg=+0.0616  max=+0.2794  min=+0.0040  std=0.0655  n=13
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      label     score   agree   disag   engag   d_lit  rt  bar                                       span
       t4.1   +0.2794    1380     250    1630    3762  ok               ███████████████████████      'What is a geometric'
       t4.2   +0.0819     890     493    1383    3822  ok               ███████                      ' is a geometric shape'
       t4.3   +0.0412     800     539    1339    3648  ok               ███                          ' a geometric shape that'
       t4.4   +0.0040     832     676    1508    4109  ok                                            ' geometric shape that uses'
       t4.5   +0.0399     826     545    1371    3792  ok               ███                          ' shape that uses complex'
       t4.6   +0.0396     847     582    1429    3818  ok               ███                          ' that uses complex numbers'
       t4.7   +0.0469     807     546    1353    3747  ok               ████                         ' uses complex numbers to'
       t4.8   +0.0301     824     595    1419    3744  ok               ███                          ' complex numbers to describe'
       t4.9   +0.0389     828     545    1373    3763  ok               ███                          ' numbers to describe its'
      t4.10   +0.0260     768     555    1323    3733  ok               ██                           ' to describe its convex'
      t4.11   +0.0485     823     533    1356    3745  ok               ████                         ' describe its convex cur'
      t4.12   +0.0566     844     505    1349    3747  ok               █████                        ' its convex curv'
      t4.13   +0.0684     903     547    1450    4063  ok               ██████                       ' convex curvatures'

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  5-TOKEN    avg=+0.1109  max=+0.3186  min=+0.0517  std=0.0667  n=12
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      label     score   agree   disag   engag   d_lit  rt  bar                                       span
       t5.1   +0.3186    1672     205    1877    3992  ok               ███████████████████████████  'What is a geometric shape'
       t5.2   +0.1307    1147     450    1597    4104  ok               ███████████                  ' is a geometric shape that'
       t5.3   +0.0939    1075     499    1574    3947  ok               ████████                     ' a geometric shape that uses'
       t5.4   +0.0517    1064     611    1675    4289  ok               ████                         ' geometric shape that uses complex'
       t5.5   +0.0750    1039     501    1540    4116  ok               ██████                       ' shape that uses complex numbers'
       t5.6   +0.0896    1065     512    1577    4027  ok               ████████                     ' that uses complex numbers to'
       t5.7   +0.0814    1030     528    1558    4081  ok               ███████                      ' uses complex numbers to describe'
       t5.8   +0.0847    1096     527    1623    4112  ok               ███████                      ' complex numbers to describe its'
       t5.9   +0.0823    1054     508    1562    4009  ok               ███████                      ' numbers to describe its convex'
      t5.10   +0.0809    1048     520    1568    4031  ok               ███████                      ' to describe its convex cur'
      t5.11   +0.1008    1094     492    1586    4125  ok               ████████                     ' describe its convex curv'
      t5.12   +0.1409    1135     402    1537    4062  ok               ████████████                 ' its convex curvatures'

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  SUMMARY: Mean score by token n-gram level
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

     nt  n_spans       mean     median        std        mad        max        min  bar
    ───────────────────────────────────────────────────────────────────────────────────────────────
      1       16    -0.0976    -0.1194     0.0644     0.0305    +0.0449    -0.1583       ░░░░░░░░                           
      2       15    -0.0369    -0.0632     0.0710     0.0238    +0.1747    -0.1008            ░░░                           
      3       14    +0.0081    -0.0164     0.0682     0.0078    +0.2355    -0.0510               █                          
      4       13    +0.0616    +0.0412     0.0655     0.0164    +0.2794    +0.0040               █████                      
      5       12    +0.1109    +0.0872     0.0667     0.0140    +0.3186    +0.0517               █████████                  

  L-MOMENT CHARACTERIZATION PER LEVEL
  ────────────────────────────────────────────────────────────────────────────────
     nt     λ₁(loc)   λ₂(scale)    τ₃(skew)    τ₄(kurt)  shape
    ──────────────────────────────────────────────────────────────────────
      1     -0.0976      0.0340     +0.4168     +0.2284  right-skewed, heavy-tailed
      2     -0.0369      0.0339     +0.5448     +0.4399  right-skewed, heavy-tailed
      3     +0.0081      0.0292     +0.6403     +0.6064  right-skewed, heavy-tailed
      4     +0.0616      0.0277     +0.5747     +0.6474  right-skewed, heavy-tailed
      5     +0.1109      0.0304     +0.5812     +0.5667  right-skewed, heavy-tailed

  Score increases with token n-gram level: YES  (median: YES)

  LEVEL DELTAS
  ──────────────────────────────────────────────────
  1-token -> 2-token:  +0.0607  (+62.2%)
  2-token -> 3-token:  +0.0450  (+121.8%)
  3-token -> 4-token:  +0.0536  (+665.3%)
  4-token -> 5-token:  +0.0492  (+79.9%)

  WITHIN-LEVEL VARIANCE
  ────────────────────────────────────────────────────────────
  1-token:  spread=0.2031  cv=0.66  cv_mad=0.26
  2-token:  spread=0.2756  cv=1.92  cv_mad=0.38
  3-token:  spread=0.2864  cv=8.47  cv_mad=0.48
  4-token:  spread=0.2754  cv=1.06  cv_mad=0.40
  5-token:  spread=0.2669  cv=0.60  cv_mad=0.16

  SCORE vs TOKEN N-GRAM LEVEL (all 70 documents)
   +0.3186 │                                                           ●  ← t5.1
   +0.2868 │                                            ●                 ← t4.1
   +0.2550 │                             ●                                ← t3.1
   +0.2232 │                                                            
   +0.1914 │              ●                                               ← t2.1
   +0.1596 │                                                           ●  ← t5.12
   +0.1279 │                                                           ●  ← t5.11
   +0.0961 │              ●                             ●              ●  ← t5.10
   +0.0643 │●                            ●              ●              ●  ← t5.4
   +0.0325 │                             ●              ●                 ← t4.10
   +0.0007 │●             ●              ●                                ← t3.13
   -0.0311 │              ●              ●                                ← t3.4
   -0.0629 │●             ●                                               ← t2.13
   -0.0947 │●             ●                                               ← t2.4
   -0.1265 │●                                                             ← t1.12
   -0.1583 │●                                                             ← t1.4
           └────────────────────────────────────────────────────────────
            1.00                                                    5.00
                                 token n-gram level                     

  Spearman rho (level -> mean score):   +1.0000
  Spearman rho (level -> median score): +1.0000
  Spearman rho (per-doc):               +0.7882
  Monotonic (mean): yes  (median): yes


────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Running: idf  [ok]  (0.9s)  d=+2.3120 (huge)  d=+2.3120  rd=+1.1088  target_mu=+0.1863  filler_mu=-0.0947

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  IDF DISCRIMINATION ANALYSIS
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

  QUERY: "Kubernetes pod autoscaling"     Corpus: 6 documents
  QUERY VECTOR: active=3437 (21.0%)

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RANKED SCORES
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            label     group       geo       eng   agree   disag   d_lit  bar
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
      exact_match    target   +0.3572   +0.8424    1672     143    6659            ██████████████████████████████
       near_match    target   +0.0154   +0.0686     623     543    5603            █                             
            cloud   related   -0.0669   -0.1965     464     691    5914      ░░░░░░                              
          related   related   -0.0722   -0.2165     438     680    5297      ░░░░░░                              
            vague    filler   -0.0767   -0.1871     480     701    5619     ░░░░░░░                              
        unrelated    filler   -0.1127   -0.2479     455     755    6121  ░░░░░░░░░░                              

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  GROUP DISTRIBUTIONS
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       group    n      mean    median       std       mad       min       max
  ────────────────────────────────────────────────────────────────────────────────
      target    2   +0.1863   +0.1863    0.1709    0.2534   +0.0154   +0.3572
     related    2   -0.0695   -0.0695    0.0026    0.0039   -0.0722   -0.0669
      filler    2   -0.0947   -0.0947    0.0180    0.0267   -0.1127   -0.0767

  L-MOMENTS PER GROUP
  ──────────────────────────────────────────────────────────────────────
      target:  λ₁=+nan  λ₂=nan  τ₃=+nan  τ₄=+nan
     related:  λ₁=+nan  λ₂=nan  τ₃=+nan  τ₄=+nan
      filler:  λ₁=+nan  λ₂=nan  τ₃=+nan  τ₄=+nan

  PAIRWISE GAPS
  ────────────────────────────────────────────────────────────
      exact_match -> near_match       gap=+0.3419  ███████████████     
       near_match -> cloud            gap=+0.0823  ████                
            cloud -> related          gap=+0.0053                      
          related -> vague            gap=+0.0046                      
            vague -> unrelated        gap=+0.0360  ██                  

  Cohen's d (target vs filler):  +2.3120  (huge)
  Robust d  (target vs filler):  +1.1088  (large)


────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Running: sparse  [ok]  (1.0s)  short_above=yes  bias_reduced=no  short:rank_imp=+0,gap=+0.3767  medium:rank_imp=+0,gap=+0.3452  long:rank_imp=+0,gap=+0.1745  bias_reduced=no  rho_geo=-0.6000  rho_eng=-0.4857

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  NORMALIZATION EFFECTIVENESS ANALYSIS
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

  QUERY: "photosynthesis in chloroplasts converts light energy into chemical energy"
  N-gram: 5     All relevant docs are supra-ngram (have n-gram signal)
  Test: does normalization give shorter relevant docs a fair ranking
        against longer off-topic filler?

  Length ratio (long/short relevant): 5.3x tokens, 6.9x n-gram windows

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SIDE-BY-SIDE: GEOMETRIC vs ENGAGED
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                 label     group    tier   toks  ng_win       geo       eng   d_lit  geo bar               |eng bar               
  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        short_relevant  relevant   short     15      11   +0.3118   +0.7656    5819       █████████████████|   ███████████████████
       medium_relevant  relevant  medium     38      34   +0.2802   +0.6987    7720       ███████████████  |   █████████████████  
         long_relevant  relevant    long     80      76   +0.1096   +0.2683    9037       ██████           |   ███████            
         long_filler_1    filler    long     63      59   -0.0650   -0.0938    8712   ░░░░                 | ░░                   
         long_filler_3    filler    long     56      52   -0.0753   -0.1229    8584   ░░░░                 |░░░                   
         long_filler_2    filler    long     65      61   -0.0873   -0.1213    8769  ░░░░░                 |░░░                   

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PER-TIER RANKING vs FILLER
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Each relevant doc (all supra-ngram) tested against 3 long filler docs.
  N-gram windows = max(0, tokens - 5 + 1)

      tier               label   toks  ng_win  geo_rank  eng_rank  rank_imp   geo_gap   eng_gap   gap_imp  above_filler
  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     short      short_relevant     15      11         1         1        +0   +0.3767   +0.8594   -0.4827  geo=yes eng=yes
    medium     medium_relevant     38      34         2         2        +0   +0.3452   +0.7925   -0.4474  geo=yes eng=yes
      long       long_relevant     80      76         3         3        +0   +0.1745   +0.3621   -0.1875  geo=yes eng=yes

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LENGTH BIAS
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Spearman rho (doc_lit vs score) — lower magnitude = less length-biased:

                          metric   geometric     engaged
  ────────────────────────────────────────────────────────────
              rho(length, score)     -0.6000     -0.4857
            length bias reduced?          no

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  OVERALL SEPARATION (relevant vs filler)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                          metric   geometric     engaged
  ────────────────────────────────────────────────────────────
                         Cohen d     +4.9056     +4.4209
              Robust d (med/MAD)     +7.5969     +8.2662

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STRUCTURAL FEATURES
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Length-score correlation not reduced by normalization (geo rho=-0.6000 vs eng rho=-0.4857).


────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Running: specificity  [ok]  (1.0s)  margin=+0.1214  opacity=0.93  margin=+0.1214  MI=0.932  opacity=0.93  norm=1.02x

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  SHARED INFORMATION TRANSFER FUNCTION
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

  QUERY: "eigenvalue decomposition of symmetric positive definite matrices"
  QUERY VECTOR: active=4502 (27.5%)
  Query tokens: 9

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PER-DOCUMENT ANALYSIS (sorted by score)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            label  shared       geo   agree   disag     lit      norm  bar
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
            exact       6   +0.1648    1588     653    7406    2574.0                ██████████████████████████  [6 tokens]
      broad_eigen       0   +0.0434     990     778    6279    2573.5                ███████                     [BLIND]
      field_level       2   -0.0276     809     917    6602    2577.5            ░░░░                            [2 tokens]
    super_generic       0   -0.0643     672     856    5638    2570.5      ░░░░░░░░░░                            [BLIND]
     broad_decomp       1   -0.0697     787    1044    6800    2625.4     ░░░░░░░░░░░                            [1 tokens]
        unrelated       1   -0.0843     744     997    6395    2582.5  ░░░░░░░░░░░░░░                            [1 tokens]

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TRANSFER FUNCTION
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Score as a function of shared token count.

    k (shared)  n_docs  mean_score  bar
  ──────────────────────────────────────────────────
             0       2     -0.0104        ░             
             1       2     -0.0770   ░░░░░░             
             2       1     -0.0276       ░░             
             6       1     +0.1648         █████████████

  Inflection region: k=2 -> k=?  (jump=+0.1924)
  Below k=3: mean score is near or below noise floor.
  Above k=2: score rises sharply with additional shared tokens.

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INFORMATION-THEORETIC MEASURES
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  H(score):                     -1.4715 bits  (marginal score entropy)
  H(shared_count):               1.9183 bits  (overlap distribution entropy)
  H(score | shared_count):      -2.4035 bits  (residual uncertainty)
  I(shared_count; score):        0.9320 bits  (information overlap provides about score)

  Spearman rho:                +0.2571
  Lexical opacity (1 - rho^2): 0.9339
  Interpretation: 93% of score variance is driven by
  non-lexical features (n-gram position, IDF weighting, composition).

  CONDITIONAL ENTROPY by overlap level
  ──────────────────────────────────────────────────
  k= 0:  H(score|k)= -2.1674  n=2
  k= 1:  H(score|k)= -5.0432  n=2
  k= 2:  H(score|k)=    ---  n=1
  k= 6:  H(score|k)=    ---  n=1

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SEPARATION METRICS
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Exact match is rank-1: yes
  Exact-match margin (rank-1 - rank-2): +0.1214
  Overlap docs (shared > 0):   n=4  mean=-0.0042
  Disjoint docs (shared = 0):  n=2  mean=-0.0104
  Group gap (overlap - disjoint):       +0.0063

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NORMALIZATION DISTORTION
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  score denom = sqrt(lit_uni * q_lit_uni)
  ────────────────────────────────────────────────────────────
      super_generic  lit=  2571  norm_denom=  2570.5
        broad_eigen  lit=  2577  norm_denom=  2573.5
              exact  lit=  2578  norm_denom=  2574.0
        field_level  lit=  2585  norm_denom=  2577.5
          unrelated  lit=  2595  norm_denom=  2582.5
       broad_decomp  lit=  2682  norm_denom=  2625.4

  Raw sqrt(lit) ratio:  1.02x
  Norm denom ratio:     1.02x

  SCORE vs SHARED TOKEN COUNT
   +0.1648 │                                                           ●  ← exact
   +0.1482 │                                                            
   +0.1316 │                                                            
   +0.1150 │                                                            
   +0.0984 │                                                            
   +0.0818 │                                                            
   +0.0652 │                                                            
   +0.0486 │●                                                             ← broad_eigen
   +0.0319 │                                                            
   +0.0153 │                                                            
   -0.0013 │                                                            
   -0.0179 │                   ●                                          ← field_level
   -0.0345 │                                                            
   -0.0511 │●                                                             ← super_generic
   -0.0677 │         ●                                                    ← broad_decomp
   -0.0843 │         ●                                                    ← unrelated
           └────────────────────────────────────────────────────────────
            0.00                                                    6.00
                                shared query tokens                     

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Running: noise  [ok]  (1.2s)  d=+2.9468 (huge)  d=+2.9468  rd=+1.1877  margin=-0.0040

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  NOISE FLOOR ANALYSIS
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

  QUERY: "quantum entanglement between photon pairs"
  Corpus: 3 signal + 8 noise

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RANKED RESULTS
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                 label     group       geo   agree   disag   d_lit  bar
  ────────────────────────────────────────────────────────────────────────────────────────────────────
              relevant    signal   +0.1992     950     277    6972          ████████████████████████████████
       highly_relevant    signal   +0.1174     756     407    7298          ███████████████████             
               noise_5     noise   -0.0141     576     599    8220        ░░                                 <-
               noise_1     noise   -0.0171     563     644    8832       ░░░                                 <-
            same_field    signal   -0.0180     508     550    6370       ░░░                                
               noise_2     noise   -0.0205     551     677    9142       ░░░                                 <-
               noise_0     noise   -0.0238     540     604    7565      ░░░░                                 <-
               noise_7     noise   -0.0297     572     660    9283     ░░░░░                                 <-
               noise_4     noise   -0.0302     542     630    9052     ░░░░░                                 <-
               noise_3     noise   -0.0449     531     669    8691   ░░░░░░░                                 <-
               noise_6     noise   -0.0502     535     662    8454  ░░░░░░░░                                 <-

  GROUP STATISTICS
  ──────────────────────────────────────────────────────────────────────
    signal: mu=+0.0995  med=+0.1174  sigma=0.0896  mad=0.1213  [-0.0180, +0.1992]  █▄ 
            L-mom: λ₁=+nan  λ₂=nan  τ₃=+nan  τ₄=+nan
     noise: mu=-0.0288  med=-0.0267  sigma=0.0121  mad=0.0118  [-0.0502, -0.0141]  ▅▇▆▁▄█ ▄
            L-mom: λ₁=-0.0288  λ₂=0.0076  τ₃=-0.2166  τ₄=+0.0671

  Separation margin (signal_min - noise_max): -0.0040
  Cohen's d (signal vs noise):                +2.9468  (huge)
  Robust d  (signal vs noise):                +1.1877  (large)


────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Running: duplicate  [??]  (0.9s)  rho=+0.1429  rho=+0.1429  r=+0.0718  jump=0.0300  iso=+0.0339

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  ENCODING SENSITIVITY PROFILE
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

  QUERY: "gradient descent optimization for neural networks"
  N-gram: 5     Base: "Gradient descent is an iterative optimization algorithm used to minimi..."

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  VARIANT SCORES
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                 label  retention       geo   agree   disag   d_lit  bar
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
              original      1.000   -0.0090     528     502    7045                    ░░░░░                 
          period_added      1.000   -0.0390     474     529    7191     ░░░░░░░░░░░░░░░░░░░░                 
             ws_padded      0.952   -0.0118     517     504    7221                   ░░░░░░                 
          synonym_swap      0.856   -0.0437     474     549    7190  ░░░░░░░░░░░░░░░░░░░░░░░                 
       phrase_collapse      0.808   -0.0192     467     528    7048               ░░░░░░░░░░                 
              reversed      0.212   -0.0252     476     508    7053            ░░░░░░░░░░░░░                 
             unrelated        ---   +0.0339     534     447    6843                         █████████████████  <- control

  RETENTION-ORDERED RANKING
  ──────────────────────────────────────────────────────────────────────
  1.             original  ret=1.000  geo=-0.0090
  2.         period_added  ret=1.000  geo=-0.0390  ^! ws_padded
  3.            ws_padded  ret=0.952  geo=-0.0118  v synonym_swap
  4.         synonym_swap  ret=0.856  geo=-0.0437  ^! phrase_collapse
  5.      phrase_collapse  ret=0.808  geo=-0.0192  v reversed
  6.             reversed  ret=0.212  geo=-0.0252

  RETENTION vs SCORE
   -0.0090 │                                                           ●  ← original
   -0.0113 │                                                       ●      ← ws_padded
   -0.0136 │                                                            
   -0.0160 │                                                            
   -0.0183 │                                            ●                 ← phrase_collapse
   -0.0206 │                                                            
   -0.0229 │                                                            
   -0.0252 │●                                                             ← reversed
   -0.0275 │                                                            
   -0.0299 │                                                            
   -0.0322 │                                                            
   -0.0345 │                                                            
   -0.0368 │                                                           ●  ← period_added
   -0.0391 │                                                            
   -0.0414 │                                                            
   -0.0437 │                                                ●             ← synonym_swap
           └────────────────────────────────────────────────────────────
            0.21                                                    1.00
                                  n-gram retention                      
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LOCAL SENSITIVITY (Lipschitz profile)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Sensitivity = |delta_score| / (1 - retention)  at each perturbation point.
  Higher values indicate the encoding amplifies that perturbation class.

                 label           class  disrupted   |delta|   L_local  bar
  ────────────────────────────────────────────────────────────────────────────────
          period_added    tokenization       0.0%    0.0300       inf  ████████████████████
          synonym_swap         lexical      14.4%    0.0347    0.2409  ████████████████████
             ws_padded    tokenization       4.8%    0.0028    0.0582  █████               
       phrase_collapse      structural      19.2%    0.0102    0.0531  ████                
              reversed      structural      78.8%    0.0162    0.0206  ██                  

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PERTURBATION CLASS SENSITIVITY
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Aggregate sensitivity by perturbation type.

           class    n    mean_L     max_L  mean_|d|  bar
  ────────────────────────────────────────────────────────────
    tokenization    2    0.0582    0.0582    0.0164  █████               
         lexical    1    0.2409    0.2409    0.0347  ████████████████████
      structural    2    0.0369    0.0531    0.0132  ███                 

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CONTINUITY ANALYSIS
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Identity-boundary jump (ret=1.0, score≠original): 0.0300
  The encoding map has a discontinuity at retention=1.0.
  Tokenization-level perturbations that preserve all n-gram content
  still shift n-gram window alignment, producing orthogonal bindings.

  Isometry constant (control score):  +0.0339
  This is the baseline correlation between independent draws from
  the encoding's output distribution — the map's noise floor.

  Variants below isometry constant:   6/6
  These perturbations produce vectors that are anti-correlated
  with the query, indicating negative curvature in those
  perturbation directions.

  Anti-correlated variants (score < 0): 5
  The n-gram promotion layer actively disagrees with the query
  when window alignment is disrupted — the encoding's output
  space has regions of negative inner product under perturbation.

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CORRELATION STRUCTURE
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Spearman rho (retention -> score): +0.1429
  Pearson r:                         +0.0718

  The encoding map f: retention -> score has a monotonic tendency
  (rho=+0.1429) with weak rank preservation.

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STRUCTURAL FEATURES
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Discontinuity at identity: jump=0.0300. The encoding amplifies tokenization-boundary perturbations ~0.1x relative to structural perturbations.
  2. Negative curvature: 6/6 perturbations produce vectors below the isometry constant (+0.0339). The n-gram promotion layer generates anti-correlated signal under window misalignment.


────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Running: topic  [ok]  (1.0s)  min_margin=+0.1229  min_margin=+0.1229  min_margin_med=+0.0057

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  TOPIC SEPARATION ANALYSIS
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

  Topics: marine, compiler, medieval     Docs/topic: 3

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  QUERY: marine
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  "coral reef fish diversity ocean ecosystem"
  Query active dims: 3057 (18.7%)

            label       topic       geo   agree   disag   d_lit  bar
  ────────────────────────────────────────────────────────────────────────────────────────────────────
         marine_0      marine   +0.2752    1018     222    7031          ████████████████████████████████ <-
         marine_1      marine   -0.0351     490     585    6709      ░░░░                                 <-
         marine_2      marine   +0.0048     527     501    6653          █                                <-
       compiler_0    compiler   -0.0691     447     634    7097  ░░░░░░░░                                
       compiler_1    compiler   -0.0428     432     557    6552     ░░░░░                                
       compiler_2    compiler   -0.0413     446     573    6550     ░░░░░                                
       medieval_0    medieval   -0.0378     529     639    7601      ░░░░                                
       medieval_1    medieval   -0.0510     465     597    6766    ░░░░░░                                
       medieval_2    medieval   -0.0351     492     605    7029      ░░░░                                

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  QUERY: compiler
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  "lexer parser abstract syntax tree code generation"
  Query active dims: 2821 (17.2%)

            label       topic       geo   agree   disag   d_lit  bar
  ────────────────────────────────────────────────────────────────────────────────────────────────────
         marine_0      marine   -0.0265     432     511    7031      ░░░                                 
         marine_1      marine   -0.0423     402     531    6709    ░░░░░                                 
         marine_2      marine   -0.0579     401     560    6653  ░░░░░░░                                 
       compiler_0    compiler   -0.0110     471     489    7097        ░                                  <-
       compiler_1    compiler   +0.2873     955     154    6552         █████████████████████████████████ <-
       compiler_2    compiler   +0.0745     583     334    6550         █████████                         <-
       medieval_0    medieval   -0.0579     434     577    7601  ░░░░░░░                                 
       medieval_1    medieval   -0.0510     391     521    6766   ░░░░░░                                 
       medieval_2    medieval   -0.0511     409     534    7029   ░░░░░░                                 

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  QUERY: medieval
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  "feudal lords vassals medieval European society"
  Query active dims: 4455 (27.2%)

            label       topic       geo   agree   disag   d_lit  bar
  ────────────────────────────────────────────────────────────────────────────────────────────────────
         marine_0      marine   -0.0825     800    1020    7031  ░░░░░░░                                 
         marine_1      marine   -0.0452     814     946    6709     ░░░░                                 
         marine_2      marine   -0.0428     780     952    6653      ░░░                                 
       compiler_0    compiler   -0.0608     820    1006    7097    ░░░░░                                 
       compiler_1    compiler   -0.0530     752     953    6552     ░░░░                                 
       compiler_2    compiler   -0.0438     774     917    6550      ░░░                                 
       medieval_0    medieval   +0.4198    2198     360    7601         █████████████████████████████████ <-
       medieval_1    medieval   -0.0637     770     943    6766    ░░░░░                                  <-
       medieval_2    medieval   -0.0394     856     986    7029      ░░░                                  <-

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  CROSS-TOPIC SCORE MATRIX
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

      query / doc        marine      compiler      medieval      margin
  ─────────────────────────────────────────────────────────────────
           marine     +0.0816 *     -0.0511       -0.0413       +0.1229
         compiler     -0.0422       +0.1169 *     -0.0533       +0.1592
         medieval     -0.0568       -0.0525       +0.1056 *     +0.1581

  DIAGONAL DOMINANCE
  ──────────────────────────────────────────────────
  marine vs compiler: +0.1327  █████████████████   
  marine vs medieval: +0.1229  ███████████████     
  compiler vs marine: +0.1592  ████████████████████
  compiler vs medieval: +0.1703  █████████████████████
  medieval vs marine: +0.1624  ████████████████████
  medieval vs compiler: +0.1581  ████████████████████

  Minimum dominance margin (mean):   +0.1229
  Minimum dominance margin (median): +0.0057

  L-MOMENTS (own-topic scores under own query)
  ──────────────────────────────────────────────────────────────────────
      marine:  λ₁=+nan  λ₂=nan  τ₃=+nan  τ₄=+nan
    compiler:  λ₁=+nan  λ₂=nan  τ₃=+nan  τ₄=+nan
    medieval:  λ₁=+nan  λ₂=nan  τ₃=+nan  τ₄=+nan


────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Running: density  [ok]  (1.0s)  H_rate=+5.1bits/tok  CV=0.407  norm=1.01x  CV=0.407  H_distort=0.542  supra_cv=0.201

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  ENCODING ENTROPY RATE ANALYSIS
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

  HDC dims: 16384     N-gram: 5     Documents: 7

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PER-DOCUMENT ENCODING
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                     label  tokens        regime  active   density   H_total  bits/tok   polarity  bar
  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
                    1_word       1     sub-ngram    3095    0.1889     14549   14549.1     0.5115  ███████             
                   2_words       2     sub-ngram    3313    0.2022     15213    7606.5     0.4983  ███████             
                   4_words       4     sub-ngram    3744    0.2285     16448    4112.0     0.4952  ████████            
                  12_words      12   supra-ngram    5494    0.3353     20572    1714.3     0.4971  ████████████        
                  sentence      20   supra-ngram    6582    0.4017     22506    1125.3     0.4979  ██████████████      
                 paragraph      58   supra-ngram    8740    0.5334     25071     432.3     0.5007  ███████████████████ 
           multi_paragraph     115   supra-ngram    9131    0.5573     25359     220.5     0.4998  ████████████████████

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ENTROPY PROFILE
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Per-dimension entropy H(d) for ternary vector with density d and balanced polarity:
  H_dim = -p+ log2(p+) - p- log2(p-) - p0 log2(p0)

                     label  tokens   density     H_dim   H_total  bar
  ────────────────────────────────────────────────────────────────────────────────
                    1_word       1    0.1889    0.8880     14549  ███████████         
                   2_words       2    0.2022    0.9285     15213  ████████████        
                   4_words       4    0.2285    1.0039     16448  █████████████       
                  12_words      12    0.3353    1.2556     20572  ████████████████    
                  sentence      20    0.4017    1.3737     22506  ██████████████████  
                 paragraph      58    0.5334    1.5302     25071  ████████████████████
           multi_paragraph     115    0.5573    1.5478     25359  ████████████████████

  MARGINAL ENTROPY RATE (dH/d_tokens between consecutive documents)
  ──────────────────────────────────────────────────────────────────────
                      from                        to    dt        dH        rate
  ────────────────────────────────────────────────────────────────────────────────
                    1_word                   2_words     1      +664     +664.01 bits/tok
                   2_words                   4_words     2     +1235     +617.53 bits/tok
                   4_words                  12_words     8     +4124     +515.44 bits/tok
                  12_words                  sentence     8     +1935     +241.84 bits/tok
                  sentence                 paragraph    38     +2565      +67.49 bits/tok
                 paragraph           multi_paragraph    57      +288       +5.06 bits/tok

  Asymptotic entropy rate (stable regime): +5.06 bits/token

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INVARIANT CHECKS
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                           check      result  detail
  ────────────────────────────────────────────────────────────────────────────────
               All docs non-zero        PASS  min_active=3095
           No saturation (< 1.0)        PASS  max_density=0.5573
        Polarity balanced (±10%)        PASS  max_dev=0.0115

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NORMALIZATION ISO-ENTROPY
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  The scoring function assumes vectors are comparable.
  Maximum entropy distortion: 0.5416  (54.2% of H_mean)

  Corpus median lit:   2584
  Norm denom ratio:    1.01x
  lit_uni range:       [2555.0, 2629.0]
  Density CV:          0.407
  Density CV (MAD):    0.589
  Entropy CV:          0.213
  Density range:       [0.1889, 0.5573]

  Density L-moments:   λ₁=+0.3496  λ₂=0.0924  τ₃=+0.1543  τ₄=-0.2103
  Entropy L-moments:   λ₁=+19959.8360  λ₂=2771.6746  τ₃=+0.0040  τ₄=-0.2632

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  REGIME TRANSITION
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Sub-ngram (tokens < 5):   pure unigram encoding, no n-gram contribution
  Supra-ngram (tokens >= 5): unigram + m_cap-limited n-grams + promotion

  Sub-ngram:    n=3  density mean=0.2065  entropy mean=15403
  Supra-ngram:  n=4  density mean=0.4570  entropy mean=23377  density CV=0.201
  Density gap:  0.2504
  Transition zone: ~12-58 tokens

  TOKEN COUNT vs DENSITY
   +0.5573 │                             ●                             ●  ← multi_paragraph
   +0.5328 │                                                            
   +0.5082 │                                                            
   +0.4836 │                                                            
   +0.4591 │                                                            
   +0.4345 │                                                            
   +0.4099 │         ●                                                    ← sentence
   +0.3854 │                                                            
   +0.3608 │                                                            
   +0.3363 │     ●                                                        ← 12_words
   +0.3117 │                                                            
   +0.2871 │                                                            
   +0.2626 │                                                            
   +0.2380 │ ●                                                            ← 4_words
   +0.2135 │●                                                             ← 2_words
   +0.1889 │●                                                             ← 1_word
           └────────────────────────────────────────────────────────────
            1.00                                                  115.00
                                    token count                         
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STRUCTURAL FEATURES
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Entropy distortion: 0.54 (max|H_i - H_j|/H_mean). The scoring function operates on representations with unequal information content.

  Pearson r (tokens -> density): +0.8828  (non-monotonic expected due to phase transition)


────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Running: sensitivity  [ok]  (0.9s)  mu_delta=0.053026  CV=0.1139  mu_delta=0.053026  CV=0.1139  all_changed=yes

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  QUERY SENSITIVITY ANALYSIS
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

  HDC dims: 16384
  Words: quantum -> field -> theory -> predicts -> particle -> interactions -> using -> gauge -> symmetry -> principles

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INCREMENTAL ENCODING
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  active   density   ntok  w_tok     delta_H  dens bar              delta bar             word added
  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    1    2691    0.1642      2      2         ---  ███████████                                 +quantum
    2    2871    0.1752      3      1    0.064636  ████████████          ████████████████████  +field
    3    3170    0.1935      4      2    0.058929  █████████████         ██████████████████    +theory
    4    3410    0.2081      5      2    0.057983  ██████████████        ██████████████████    +predicts
    5    3741    0.2283      6      1    0.053436  ███████████████       █████████████████     +particle
    6    3999    0.2441      7      2    0.052002  ████████████████      ████████████████      +interactions
    7    4253    0.2596      8      1    0.050049  █████████████████     ███████████████       +using
    8    4520    0.2759      9      2    0.048065  ██████████████████    ███████████████       +gauge
    9    4753    0.2901     10      2    0.047211  ███████████████████   ███████████████       +symmetry
   10    4959    0.3027     11      2    0.044922  ████████████████████  ██████████████        +principles

  HAMMING DELTA STATISTICS
  ──────────────────────────────────────────────────
  Mean:    0.053026
  Median:  0.052002
  Std:     0.006041
  MAD:     0.007104
  CV:      0.1139
  CV(MAD): 0.1366
  Range:   [0.044922, 0.064636]
  L-mom:   λ₁=+0.0530  λ₂=0.0038  τ₃=+0.1696  τ₄=+0.0719
  Spark:   █▅▅▃▂▂▁  

  All additions produced change: yes
  Mean Hamming delta per word:   0.053026
  CV: 0.1139  (stable)

  NOTE: ntok = cumulative token count, w_tok = tokens added by this word.
  The encoder operates on tokens, not words. Multi-subword words
  contribute more tokens per step than single-token words.


────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Running: retrieval  [ok]  (1.0s)  d=+5.1022 (huge)  d=+5.1022  rd=+10.5206  P@1=1.00  P@4=1.00  run=4/4

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  RETRIEVAL PIPELINE ANALYSIS
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

  QUERY: "how do transformers use self-attention mechanisms for sequence modeling"
  Corpus: 4 on-topic + 6 off-topic
  QUERY VECTOR: active=4106 (25.1%)

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RANKED RESULTS
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  rank                 label           topic       geo   agree   disag   d_lit  bar
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
     1            positional              on   +0.1584    1148     528    6315              ████████████████████████████ <-
     2        self_attention              on   +0.0689    1021     729    7327              ████████████                 <-
     3              bert_gpt              on   +0.0569     993     778    7423              ██████████                   <-
     4    transformers_exact              on   +0.0543    1019     728    7453              █████████                    <-
     5                  rnns      related_ml   -0.0463     715     858    6511      ░░░░░░░░                            
     6            french_rev         history   -0.0493     705     846    6436     ░░░░░░░░░                            
     7                  cnns      related_ml   -0.0504     677     823    6032     ░░░░░░░░░                            
     8             c4_plants         biology   -0.0511     593     729    5530     ░░░░░░░░░                            
     9                   gbm        other_ml   -0.0520     609     743    5518     ░░░░░░░░░                            
    10             volcanoes         geology   -0.0707     637     909    6602  ░░░░░░░░░░░░                            

  SCORE BY TOPIC GROUP
  ──────────────────────────────────────────────────────────────────────
              on: mu=+0.0846  med=+0.0629  [+0.0543, +0.1584]  n=4   ▁█ 
      related_ml: mu=-0.0483  med=-0.0483  [-0.0504, -0.0463]  n=2  █ 
         history: mu=-0.0493  med=-0.0493  [-0.0493, -0.0493]  n=1   
         biology: mu=-0.0511  med=-0.0511  [-0.0511, -0.0511]  n=1   
        other_ml: mu=-0.0520  med=-0.0520  [-0.0520, -0.0520]  n=1   
         geology: mu=-0.0707  med=-0.0707  [-0.0707, -0.0707]  n=1   

  PRECISION
  ────────────────────────────────────────
  P@1: 1.00  ████████████████████
  P@2: 1.00  ████████████████████
  P@4: 1.00  ████████████████████

  Correct run before first miss: 4/4
  Cohen's d (on vs off):         +5.1022  (huge)
  Robust d  (on vs off):         +10.5206  (huge)

  L-MOMENTS
  ────────────────────────────────────────────────────────────
        on-topic:  λ₁=+0.0846  λ₂=0.0270  τ₃=+0.8042  τ₄=+0.6301
       off-topic:  λ₁=-0.0533  λ₂=0.0044  τ₃=-0.5928  τ₄=+0.7686

  TOP RESULT
  ──────────────────────────────────────────────────
  Document:    positional
  Score:       +0.1584
  Agree ratio: 0.6850


────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Running: paraphrase  [ok]  (1.0s)  med_ret=0.736  above_filler=0/5  med_ret=0.736  med_gap=+0.0044  above_filler=0/5

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  PARAPHRASE ROBUSTNESS ANALYSIS
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

  QUERY: "machine learning algorithms for image classification"
  Pairs: 5     Filler: 3

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PER-PAIR SCORES
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                 label     exact      para   retain       gap  exact bar         |para bar
  ────────────────────────────────────────────────────────────────────────────────────────────────────
    cnn_classification   +0.3233   -0.0353   -0.109   +0.3585    ████████████████|░░                
      gradient_descent   -0.0276   -0.0320    1.160   +0.0044   ░                |░░                
        rnn_sequential   -0.0342   -0.0252    0.736   -0.0090  ░░                | ░                
     transfer_learning   +0.0076   -0.0114   -1.505   +0.0190                    | ░                
          augmentation   -0.0384   -0.0314    0.816   -0.0071  ░░                |░░                

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FILLER BASELINE
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              filler_1   -0.0334  ░░                
              filler_2   -0.0489  ░░                
              filler_3   +0.0047                    

  Best filler: +0.0047
  Paraphrases above filler: 0/5

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RETENTION STATISTICS
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Mean retention:    0.2197
  Median retention:  0.7365
  MAD:               0.6277
  Range:             [-1.5046, 1.1598]
  L-moments:         λ₁=+0.2197  λ₂=0.6254  τ₃=-0.4588  τ₄=+0.2603

  Exact scores:      med=-0.0276  L-mom: λ₁=+0.0461  λ₂=0.0765  τ₃=+0.8513  τ₄=+0.7269
  Para scores:       med=-0.0314  L-mom: λ₁=-0.0271  λ₂=0.0055  τ₃=+0.4869  τ₄=+0.3763
  Median gap:        +0.0044

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STRUCTURAL FEATURES
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Only 0/5 paraphrases outscore the best filler (+0.0047).


────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Running: query_length  [ok]  (0.9s)  P@1_at=7tok  med_gap=+0.2255  P@1_at=7tok  gaps=[+0.000,+0.665]

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  QUERY LENGTH RETRIEVAL CURVE
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

  Target: "Transformer neural networks use multi-head self-attention mechanisms t..."
  Filler: 5 unrelated docs
  Query words: transformer → attention → mechanism → sequence → modeling → neural → network → language...

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RETRIEVAL BY QUERY LENGTH
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  words  tokens   P@1    target    filler       gap  gap bar
  ────────────────────────────────────────────────────────────────────────────────
      1       2     ✗   +0.0000   +0.0000   +0.0000                      
      2       3     ✗   +0.0000   +0.0000   +0.0000                      
      3       4     ✗   +0.0000   +0.0000   +0.0000                      
      4       5     ✗   +0.0000   +0.0000   +0.0000                      
      5       6     ✗   +0.0000   +0.0000   +0.0000                      
      6       7     ✓   +0.0969   -0.0042   +0.1011  ███                 
      7       8     ✓   +0.0969   -0.0042   +0.1011  ███                 
      8       9     ✓   +0.1594   -0.0296   +0.1890  ██████              
      9      10     ✓   +0.2137   -0.0483   +0.2620  ████████            
     10      13     ✓   +0.3890   -0.0609   +0.4498  ██████████████      
     11      16     ✓   +0.4887   -0.0642   +0.5529  █████████████████   
     12      17     ✓   +0.4976   -0.0705   +0.5681  █████████████████   
     13      20     ✓   +0.5426   -0.0719   +0.6144  ██████████████████  
     14      21     ✓   +0.5616   -0.0715   +0.6331  ███████████████████ 
     15      22     ✓   +0.5790   -0.0707   +0.6496  ████████████████████
     16      23     ✓   +0.5950   -0.0702   +0.6652  ████████████████████

  First P@1=1.0 at 7 tokens
  Gap L-moments: λ₁=+0.2991  λ₂=0.1595  τ₃=+0.0800  τ₄=-0.2158

  QUERY LENGTH vs TARGET-FILLER GAP
   +0.6652 │                                                     ●  ●  ●  ← 16w
   +0.6208 │                                                  ●           ← 13w
   +0.5765 │                                       ●  ●                   ← 12w
   +0.5321 │                                                            
   +0.4878 │                              ●                               ← 10w
   +0.4434 │                                                            
   +0.3991 │                                                            
   +0.3548 │                                                            
   +0.3104 │                                                            
   +0.2661 │                      ●                                       ← 9w
   +0.2217 │                   ●                                          ← 8w
   +0.1774 │                                                            
   +0.1330 │              ● ●                                             ← 7w
   +0.0887 │                                                            
   +0.0443 │                                                            
   +0.0000 │● ●  ●  ●  ●                                                  ← 5w
           └────────────────────────────────────────────────────────────
            2.00                                                   23.00
                                    query tokens                        

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Running: corpus_scale  [ok]  (14.4s)  d@500=+10.3671  rho=+1.0000  d@500=+10.3671  rd=+1.8037  rho_scale=+1.0000

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  CORPUS SCALE ANALYSIS
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

  QUERY: "quantum entanglement between photon pairs"
  Signal: 3 docs (fixed)     Noise: 5..500 (scaled)

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DISCRIMINATION BY CORPUS SIZE
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   noise   total    Cohen d   robust d    margin   noi_med   noi_p99  d bar
  ──────────────────────────────────────────────────────────────────────────────────────────
       5       8    +2.3884    +3.2210   +0.0213   -0.0415   -0.0006  █████               
      10      13    +3.2797    +2.8873   +0.0377   -0.0195   -0.0021  ██████              
      25      28    +4.4606    +1.6076   +0.0089   -0.0109   +0.0247  █████████           
      50      53    +5.5890    +1.7357   -0.0270   -0.0088   +0.0484  ███████████         
     100     103    +6.8001    +1.8389   -0.0350   -0.0044   +0.0384  █████████████       
     250     253    +9.2942    +1.5680   -0.0011   -0.0014   +0.0365  ██████████████████  
     500     503   +10.3671    +1.8037   +0.0127   -0.0013   +0.0343  ████████████████████

  Scale-d correlation (Spearman rho): +1.0000
  Interpretation: d is stable across corpus scale.

  NOISE DISTRIBUTION L-MOMENTS AT LARGEST SCALE
  λ₁=-0.0008  λ₂=0.0066  τ₃=+0.0563  τ₄=+0.1675

  CORPUS SIZE vs COHEN'S d
  +10.3671 │                                                           ●  ← n=500
   +9.8352 │                                                            
   +9.3032 │                             ●                                ← n=250
   +8.7713 │                                                            
   +8.2394 │                                                            
   +7.7075 │                                                            
   +7.1756 │           ●                                                  ← n=100
   +6.6437 │                                                            
   +6.1118 │     ●                                                        ← n=50
   +5.5799 │                                                            
   +5.0479 │                                                            
   +4.5160 │  ●                                                           ← n=25
   +3.9841 │                                                            
   +3.4522 │●                                                             ← n=10
   +2.9203 │                                                            
   +2.3884 │●                                                             ← n=5
           └────────────────────────────────────────────────────────────
            5.00                                                  500.00
                                  noise documents                       
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STRUCTURAL FEATURES
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. 99th percentile noise approaches signal at 4 scale(s).


────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Running: adversarial  [ok]  (1.0s)  margin=+0.1095  rank=1  margin=+0.1095  rank=1  decoy_d=+2.6966

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  ADVERSARIAL VOCABULARY ANALYSIS
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

  QUERY: "neural network training with backpropagation gradient updates"
  Target: 1     Decoys: 4 (shared vocabulary, wrong context)
  Filler: 2 (no shared vocabulary)

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RANKED SCORES
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                 label     group       geo   agree   disag  bar
  ────────────────────────────────────────────────────────────────────────────────
            true_match    target   +0.1367    1289     571               ███████████████████████████
           neuro_decoy     decoy   +0.0273    1094     950               █████                      
          hiking_decoy     decoy   +0.0254    1054     915               █████                      
         fishing_decoy     decoy   -0.0061     875     832              ░                           
       corporate_decoy     decoy   -0.0326     828     885        ░░░░░░░                           
           unrelated_2    filler   -0.0474     824     943     ░░░░░░░░░░                           
             unrelated    filler   -0.0628     819    1027  ░░░░░░░░░░░░░                           

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ADVERSARIAL RESISTANCE
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Target score:          +0.1367
  Best decoy score:      +0.0273
  Target-decoy margin:   +0.1095
  Target rank:           1

  Decoy scores:  med=+0.0096  MAD=0.0247
  Filler scores: med=-0.0551  MAD=0.0114
  Decoy L-moments: λ₁=+0.0035  λ₂=0.0176  τ₃=-0.3496  τ₄=-0.4916

  Decoy vs filler d:     +2.6966 (huge)
  Decoy vs filler rd:    +2.6165

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STRUCTURAL FEATURES
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Decoys score significantly above filler (d=+2.6966). Shared vocabulary inflates scores.


────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Running: mmr  [ok]  (1.2s)  raw=2  mmr=3  gain=+1  raw_clusters=2  mmr_clusters=3  gain=+1

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  MMR DIVERSITY ANALYSIS
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

  QUERY: "how does photosynthesis convert sunlight into chemical energy"
  Clusters: A (3 near-dupes), B (2 Calvin cycle), C (1 chlorophyll)
  Filler: 3     K=4     MMR λ=0.7

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RAW RANKING (top-4)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1.     photo_a3  cluster=A  score=+0.2344
  2.     photo_a1  cluster=A  score=+0.2221
  3.     photo_a2  cluster=A  score=+0.1068
  4.    calvin_b1  cluster=B  score=+0.0062
  Unique relevant clusters: 2

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MMR RANKING (top-4)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1.     photo_a3  cluster=A  score=+0.2344
  2.     photo_a1  cluster=A  score=+0.2221
  3.    calvin_b1  cluster=B  score=+0.0062
  4.    chloro_c1  cluster=C  score=-0.0565
  Unique relevant clusters: 3

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DIVERSITY COMPARISON
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Raw unique clusters:   2
  MMR unique clusters:   3
  Diversity gain:        +1

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ALL SCORES
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         label  cluster       geo  bar
  ────────────────────────────────────────────────────────────
      photo_a3        A   +0.2344             █████████████████████████████
      photo_a1        A   +0.2221             ████████████████████████████ 
      photo_a2        A   +0.1068             █████████████                
     calvin_b1        B   +0.0062             █                            
     chloro_c1        C   -0.0565      ░░░░░░░                             
      filler_3        X   -0.0577      ░░░░░░░                             
     calvin_b2        B   -0.0635     ░░░░░░░░                             
      filler_1        X   -0.0848  ░░░░░░░░░░░                             
      filler_2        X   -0.0882  ░░░░░░░░░░░                             


────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Running: code_prose  [ok]  (1.1s)  cross=4/6  p2c=1  c2p=3  cross=4/6  p2c_gap=-0.0283  c2p_gap=+0.0419

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  CODE vs PROSE CROSS-DOMAIN ANALYSIS
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

  Topics: bsearch, linkedlist, quicksort
  Each topic has a prose doc and a code doc.

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PROSE QUERY → CODE DOCUMENT
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         topic  same_prose  cross_code  best_other   cross_gap  found?
  ────────────────────────────────────────────────────────────────────────────────
       bsearch     +0.2853     -0.0677     -0.0677     -0.0943  ✗
    linkedlist     +0.2783     -0.0283     -0.0283     -0.0283  ✗
     quicksort     +0.2740     +0.0814     +0.0814     +0.1197  ✓

  Prose→Code P@1: 1/3
  Median cross gap: -0.0283

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CODE QUERY → PROSE DOCUMENT
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         topic   same_code  cross_prose  best_other   cross_gap  found?
  ────────────────────────────────────────────────────────────────────────────────
       bsearch     +0.3935     +0.0596     +0.0596     +0.0336  ✓
    linkedlist     +0.1246     +0.0157     +0.0157     +0.0482  ✓
     quicksort     +0.2373     -0.0026     -0.0026     +0.0419  ✓

  Code→Prose P@1: 3/3
  Median cross gap: +0.0419

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CROSS-DOMAIN SUMMARY
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total cross-domain hits: 4/6
  Prose→Code L-moments: λ₁=+nan  λ₂=nan  τ₃=+nan  τ₄=+nan
  Code→Prose L-moments: λ₁=+nan  λ₂=nan  τ₃=+nan  τ₄=+nan


────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Running: ngram_sweep  [ok]  (12.6s)  best=n5(d=+2.21)  worst=n3(d=+2.18)  sweep=2..5  n2:d=+2.19,m=+0.167  n3:d=+2.18,m=+0.169  n4:d=+2.21,m=+0.175  n5:d=+2.21,m=+0.178

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  N-GRAM ORDER SWEEP
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

  HDC dims: 16384     Sweep: ngram=2..5
  Testing noise separation, topic selectivity, and short query threshold
  at each n-gram order to find the behavioral sweet spot.

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SCORING WEIGHTS
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ngram    α_uni     α_ng  uni bar               ng bar              
  ──────────────────────────────────────────────────────────────────────
      2    0.500    0.500  ██████████            ██████████          
      3    0.667    0.333  █████████████         ███████             
      4    0.750    0.250  ███████████████       █████               
      5    0.800    0.200  ████████████████      ████                

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NOISE SEPARATION BY N-GRAM ORDER
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Query: "quantum entanglement between photon pairs"
  Signal: 3 docs     Noise: 6 random docs

  ngram    Cohen d    margin    sig_mu    noi_mu       label  bar
  ────────────────────────────────────────────────────────────────────────────────
      2    +2.1852   +0.0070   +0.0711   -0.0217        huge  ████████████████████
      3    +2.1843   +0.0060   +0.0695   -0.0208        huge  ████████████████████
      4    +2.2122   +0.0060   +0.0690   -0.0206        huge  ████████████████████
      5    +2.2127   +0.0044   +0.0693   -0.0212        huge  ████████████████████

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TOPIC SELECTIVITY BY N-GRAM ORDER
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Topics: marine, compiler, medieval     Docs/topic: 2

  ngram  min_margin  separated                         margins  bar
  ────────────────────────────────────────────────────────────────────────────────
      2     +0.1674        yes       +0.1674  +0.2391  +0.2770  ███████████████████ 
      3     +0.1689        yes       +0.1689  +0.2329  +0.2681  ███████████████████ 
      4     +0.1746        yes       +0.1746  +0.2335  +0.2622  ████████████████████
      5     +0.1781        yes       +0.1781  +0.2356  +0.2651  ████████████████████

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SHORT QUERY THRESHOLD BY N-GRAM ORDER
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Words: quantum → field → theory → predicts → particle → interactions → using → gauge
  'positive_at' = minimum word count where target outscores all filler.

  ngram  pos_at    2w_gap  3w_gap  4w_gap  5w_gap  6w_gap  7w_gap  8w_gap
  ────────────────────────────────────────────────────────────────────────────────
      2       2    +0.201  +0.269  +0.353  +0.413  +0.458  +0.536  +0.579
      3       2    +0.201  +0.269  +0.351  +0.412  +0.453  +0.530  +0.571
      4       2    +0.201  +0.269  +0.350  +0.409  +0.449  +0.525  +0.565
      5       2    +0.201  +0.270  +0.350  +0.409  +0.449  +0.526  +0.563

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SWEEP SUMMARY
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ngram    noise_d  topic_margin  short_pos    α_uni  recommendation
  ──────────────────────────────────────────────────────────────────────
      2    +2.1852       +0.1674    2 words    0.500  ✓
      3    +2.1843       +0.1689    2 words    0.667  ✓
      4    +2.2122       +0.1746    2 words    0.750  ✓
      5    +2.2127       +0.1781    2 words    0.800  ✓

  NOISE SEPARATION vs N-GRAM ORDER
   +2.2127 │                                       ●                   ●  ← n=5
   +2.2108 │                                                            
   +2.2089 │                                                            
   +2.2070 │                                                            
   +2.2051 │                                                            
   +2.2032 │                                                            
   +2.2013 │                                                            
   +2.1995 │                                                            
   +2.1976 │                                                            
   +2.1957 │                                                            
   +2.1938 │                                                            
   +2.1919 │                                                            
   +2.1900 │                                                            
   +2.1881 │                                                            
   +2.1862 │●                                                             ← n=2
   +2.1843 │                   ●                                          ← n=3
           └────────────────────────────────────────────────────────────
            2.00                                                    5.00
                                    n-gram order                        

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  SUMMARY
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

            probe                         primary statistic    time  health
  ───────────────────────────────────────────────────────────────────────────
            ngram                               rho=+1.0000    1.0s  ok
              idf                          d=+2.3120 (huge)    0.9s  ok
           sparse          short_above=yes  bias_reduced=no    1.0s  ok*
      specificity              margin=+0.1214  opacity=0.93    1.0s  ok
            noise                          d=+2.9468 (huge)    1.2s  ok
        duplicate                               rho=+0.1429    0.9s  ??
            topic                        min_margin=+0.1229    1.0s  ok
          density             H_rate=+5.1bits/tok  CV=0.407    1.0s  ok*
      sensitivity              mu_delta=0.053026  CV=0.1139    0.9s  ok
        retrieval                          d=+5.1022 (huge)    1.0s  ok
       paraphrase           med_ret=0.736  above_filler=0/5    1.0s  ok*
     query_length              P@1_at=7tok  med_gap=+0.2255    0.9s  ok
     corpus_scale               d@500=+10.3671  rho=+1.0000   14.4s  ok*
      adversarial                    margin=+0.1095  rank=1    1.0s  ok*
              mmr                     raw=2  mmr=3  gain=+1    1.2s  ok
       code_prose                   cross=4/6  p2c=1  c2p=3    1.1s  ok
      ngram_sweep       best=n5(d=+2.21)  worst=n3(d=+2.18)   12.6s  ok

  16/17 healthy  (42.2s total)

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  ENCODING MAP CHARACTERIZATION
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════


  ──────────────────────────────────────────────────────────────────────
  Discontinuity at identity: jump=0.0300. The encoding amplifies tokenization-boundary perturbations ~0.1x relative to structural perturbations.
  Negative curvature: 6/6 perturbations produce vectors below the isometry constant (+0.0339). The n-gram promotion layer generates anti-correlated signal under window misalignment.

  CAPACITY (entropy rate)
  ──────────────────────────────────────────────────────────────────────
  Entropy distortion: 0.54 (max|H_i - H_j|/H_mean). The scoring function operates on representations with unequal information content.

  OTHER
  ──────────────────────────────────────────────────────────────────────
  [sparse] Length-score correlation not reduced by normalization (geo rho=-0.6000 vs eng rho=-0.4857).
  [paraphrase] Only 0/5 paraphrases outscore the best filler (+0.0047).
  [corpus_scale] 99th percentile noise approaches signal at 4 scale(s).
  [adversarial] Decoys score significantly above filler (d=+2.6966). Shared vocabulary inflates scores.

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  ENCODING SIGNATURE
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

  Compositional ramp:    rho=+1.0000  rho_med=+1.0000
  IDF discrimination:    d=+2.3120  rd=+1.1088
  Normalization:         short:above medium:above long:above  bias_reduced=no
  Transfer function:     MI=0.932bits  opacity=0.93
  Signal isolation:      d=+2.9468  rd=+1.1877  margin=-0.0040
  Sensitivity:           rho=+0.1429  jump=0.0300  iso=+0.0339
  Topic selectivity:     min_margin=+0.1229  min_margin_med=+0.0057
  Entropy rate:          asymptotic=+5.1bits/tok  H_distort=0.542  CV_mad=0.589
  Query resolution:      mu_delta=0.0530  med_delta=0.0520  CV=0.1139  CV_mad=0.1366
  End-to-end retrieval:  d=+5.1022  rd=+10.5206  P@1=1.00
  Paraphrase:            med_ret=0.736  above_filler=0/5
  Query length:          P@1_at=7tok  med_gap=+0.2255
  Corpus scale:          d@500=+10.3671  rho=+1.0000
  Adversarial:           margin=+0.1095  rank=1  decoy_d=+2.6966
  MMR diversity:         raw=2  mmr=3  gain=+1
  Code/prose:            cross=4/6  p2c=1  c2p=3
  N-gram sweep:          best=n5  worst=n3  n2:d=+2.19 n3:d=+2.18 n4:d=+2.21 n5:d=+2.21
========================================================================================================================
```


  Results saved to hdrag_results.txt
