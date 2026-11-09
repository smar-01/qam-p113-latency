# META across venues, 2026-09-08 (extended session)
Venues: ARCX, BATS, BATY, EDGA, EDGX, EPRL, IEXG, MEMX, XASE, XBOS, XCHI, XNAS, XNYS, XPSX. Rows: 938,674.

## Clock: ts_recv
Big moves found: 20 (>= 10.0 bps over 10 s; first 5 min after the 09:30 open and around the 16:00 close skipped).

Midpoint measures use only quotes with spread <= 25.0 bps. Core (typically tight) venues: XNAS, ARCX, BATS, EDGX, XNYS, IEXG, MEMX, XCHI.

### Biggest moves
```
 move      p0      p1  move_bps start_ny
    1 619.585 621.330    28.164 09:35:53
    2 621.340 619.470   -30.096 09:36:06
    3 619.438 617.592   -29.785 09:36:30
    4 616.882 618.430    25.086 10:02:17
    5 618.300 616.422   -30.366 10:11:28
    6 614.340 616.560    36.136 14:56:42
    7 616.750 619.035    37.049 14:56:59
    8 618.005 621.340    53.964 15:00:00
    9 620.535 622.435    30.619 15:00:50
   10 622.925 620.670   -36.200 15:01:05
   11 623.090 619.885   -51.437 15:06:17
   12 620.755 618.300   -39.549 15:06:41
   13 617.775 620.995    52.123 15:07:21
   14 620.935 623.082    34.585 15:07:32
   15 620.100 617.155   -47.492 15:09:17
   16 616.520 613.505   -48.904 15:10:16
   17 615.600 612.955   -42.966 15:10:29
   18 610.782 612.618    30.043 15:12:26
   19 610.510 613.682    51.965 15:14:43
   20 614.335 611.540   -45.496 15:22:07
```

### Who reaches the new price first (big moves)
```
venue  n_moves  n_reached_halfway  first_share  median_rank  median_lag_behind_first_us  p90_lag_behind_first_us
 XNYS     20.0               20.0         0.20          5.0                    39436.75               2755250.95
 EDGX     20.0               20.0         0.00          5.0                    51050.02               3468991.34
 MEMX     20.0               20.0         0.20          4.0                    67902.82               2665301.05
 ARCX     20.0               20.0         0.00          5.5                    84425.76               2667001.65
 BATS     20.0               20.0         0.10          4.0                    87944.96               3168988.23
 XNAS     20.0               20.0         0.15          4.5                    90059.62               2868530.55
 IEXG     20.0               19.0         0.05          8.0                   130143.97               6261802.17
 EDGA     20.0               17.0         0.15          7.0                   252442.85               5114073.64
 EPRL     20.0               19.0         0.05          7.0                   279763.97               7107986.34
 XCHI     20.0               19.0         0.05         10.0                  1321713.73               7405932.96
 XPSX     20.0               14.0         0.00         11.0                  2585739.92              14085160.39
 XBOS     20.0               13.0         0.05         11.0                  3816836.52               9101359.40
 BATY     20.0               11.0         0.00         11.0                  5465017.58              10226798.59
 XASE     20.0                3.0         0.00         13.0                 14420110.92              16955840.39
```

### Time at the inside quote (whole session vs big moves)
```
       pct_time_at_best_bid         pct_time_at_best_ask         avg_bid_behind_best_cents         
window            big_moves session            big_moves session                 big_moves  session
venue                                                                                              
ARCX                  42.60   47.27                45.79   32.80                      9.69     9.92
BATS                  24.00   20.15                19.55   11.26                     38.36    57.49
BATY                   8.59    3.92                 1.04    1.50                    822.84   816.44
EDGA                   5.05    3.92                 2.09    5.57                    229.55   487.08
EDGX                  20.91   23.66                13.29   17.42                     28.77    18.17
EPRL                   0.27    0.84                 0.12    0.58                    247.21   388.18
IEXG                   4.06   18.69                 5.19    3.93                    205.00    59.42
MEMX                   9.90   10.68                 5.32    4.43                     84.65    48.66
XASE                   0.00    1.54                 1.50    0.50                   3423.75  3697.37
XBOS                   5.83    3.54                 0.04    0.66                   1220.52  1096.50
XCHI                   1.07    2.34                 0.16    0.50                     62.44    69.05
XNAS                  63.97   44.51                69.99   50.19                      5.40    11.29
XNYS                   7.90   12.85                 2.56   33.23                     59.64    51.91
XPSX                   0.11    0.40                 0.54    0.17                   3214.88  2967.54
```

Combined NBBO locked or crossed: 23.145% of the session.

### Leadership share (reaction-latency engine)
```
        session  big_moves  session_core_venues  big_moves_core_venues
leader                                                                
XNAS      0.656      0.682                0.669                  0.693
MEMX      0.617      0.593                0.614                  0.582
BATS      0.549      0.586                0.543                  0.579
ARCX      0.457      0.518                0.449                  0.513
EDGX      0.549      0.511                0.549                  0.489
XNYS      0.305      0.301                0.283                  0.282
XCHI      0.379      0.341                0.268                  0.267
IEXG      0.293      0.283                0.239                  0.201
EPRL      0.654      0.574                  NaN                    NaN
XBOS      0.548      0.528                  NaN                    NaN
EDGA      0.499      0.529                  NaN                    NaN
BATY      0.371      0.377                  NaN                    NaN
XPSX      0.349      0.397                  NaN                    NaN
XASE      0.144      0.166                  NaN                    NaN
```

### Venue activity
```
venue  updates  share_of_all_updates  updates_per_s  updates_per_s_in_big_moves  median_spread_cents  median_spread_bps
 XNAS   276774                 0.295          4.805                      62.135                 19.0              3.076
 ARCX   154019                 0.164          2.674                      45.475                 31.0              4.973
 BATS   146250                 0.156          2.539                      29.380                 28.0              4.552
 EDGX    83988                 0.089          1.458                      29.230                 39.0              6.321
 XNYS    61125                 0.065          1.061                      13.175                 56.0              9.100
 IEXG    52755                 0.056          0.916                      12.415                147.0             23.861
 MEMX    41346                 0.044          0.718                      11.355                 50.0              8.077
 EPRL    23407                 0.025          0.406                       6.215                246.0             39.894
 XBOS    23159                 0.025          0.402                      11.385               1186.0            192.393
 XPSX    21424                 0.023          0.372                       7.480               3373.0            552.093
 EDGA    20159                 0.021          0.350                       6.660                204.0             33.012
 BATY    17742                 0.019          0.308                       6.485                729.0            118.382
 XCHI     9423                 0.010          0.164                       2.330                125.0             20.386
 XASE     7103                 0.008          0.123                       1.055               4433.0            742.341
```

## Clock: ts_event
Big moves found: 20 (>= 10.0 bps over 10 s; first 5 min after the 09:30 open and around the 16:00 close skipped).

Midpoint measures use only quotes with spread <= 25.0 bps. Core (typically tight) venues: XNAS, ARCX, BATS, EDGX, XNYS, IEXG, MEMX, XCHI.

### Biggest moves
```
 move      p0      p1  move_bps start_ny
    1 619.585 621.330    28.164 09:35:53
    2 621.340 619.470   -30.096 09:36:06
    3 619.438 617.592   -29.785 09:36:30
    4 616.882 618.430    25.086 10:02:17
    5 618.300 616.422   -30.366 10:11:28
    6 614.335 616.560    36.218 14:56:42
    7 616.750 619.035    37.049 14:56:59
    8 618.005 621.340    53.964 15:00:00
    9 620.535 622.435    30.619 15:00:50
   10 622.925 620.670   -36.200 15:01:05
   11 623.090 619.885   -51.437 15:06:17
   12 620.755 618.300   -39.549 15:06:41
   13 617.775 620.995    52.123 15:07:21
   14 620.935 623.082    34.585 15:07:32
   15 620.100 617.135   -47.815 15:09:17
   16 616.520 613.505   -48.904 15:10:16
   17 615.600 612.955   -42.966 15:10:29
   18 610.782 612.618    30.043 15:12:26
   19 610.510 613.682    51.965 15:14:43
   20 614.335 611.540   -45.496 15:22:07
```

### Who reaches the new price first (big moves)
```
venue  n_moves  n_reached_halfway  first_share  median_rank  median_lag_behind_first_us  p90_lag_behind_first_us
 XNYS     20.0               20.0         0.20          5.0                    39312.70               2755073.97
 EDGX     20.0               20.0         0.00          5.0                    51205.99               3469151.68
 ARCX     20.0               20.0         0.05          5.0                    84326.07               2666997.96
 BATS     20.0               20.0         0.05          4.0                    88011.46               3168988.90
 XNAS     20.0               20.0         0.15          4.0                    90005.44               2868397.41
 MEMX     20.0               20.0         0.15          3.5                    96751.07               2665461.12
 IEXG     20.0               19.0         0.05          8.0                   129880.83               6261562.63
 EDGA     20.0               17.0         0.20          7.0                   252585.24               5114173.73
 EPRL     20.0               19.0         0.05          7.0                   279915.81               7108018.65
 XCHI     20.0               19.0         0.05         10.0                  1321545.90               7405781.37
 XPSX     20.0               14.0         0.00         11.0                  2585682.87              14085146.97
 XBOS     20.0               13.0         0.05         11.0                  3816877.26               9101363.71
 BATY     20.0               11.0         0.00         11.0                  5465197.04              10226778.61
 XASE     20.0                3.0         0.00         13.0                 14420061.84              16955684.52
```

### Time at the inside quote (whole session vs big moves)
```
       pct_time_at_best_bid         pct_time_at_best_ask         avg_bid_behind_best_cents         
window            big_moves session            big_moves session                 big_moves  session
venue                                                                                              
ARCX                  42.60   47.27                45.79   32.80                      9.69     9.92
BATS                  23.98   20.14                19.54   11.26                     38.36    57.49
BATY                   8.58    3.92                 1.04    1.50                    822.84   816.44
EDGA                   5.05    3.92                 2.09    5.57                    229.54   487.08
EDGX                  20.90   23.66                13.30   17.41                     28.77    18.17
EPRL                   0.27    0.84                 0.12    0.58                    247.21   388.18
IEXG                   4.06   18.69                 5.19    3.93                    205.02    59.42
MEMX                   9.90   10.68                 5.32    4.44                     84.65    48.66
XASE                   0.00    1.54                 1.50    0.50                   3423.75  3697.37
XBOS                   5.83    3.54                 0.04    0.66                   1220.52  1096.50
XCHI                   1.07    2.34                 0.16    0.50                     62.44    69.05
XNAS                  63.96   44.51                70.00   50.18                      5.40    11.29
XNYS                   7.90   12.85                 2.56   33.23                     59.64    51.91
XPSX                   0.11    0.40                 0.54    0.17                   3214.88  2967.54
```

Combined NBBO locked or crossed: 23.150% of the session.

### Leadership share (reaction-latency engine)
```
        session  big_moves  session_core_venues  big_moves_core_venues
leader                                                                
XNAS      0.661      0.691                0.662                  0.690
BATS      0.548      0.553                0.553                  0.557
ARCX      0.495      0.551                0.489                  0.550
MEMX      0.528      0.522                0.531                  0.513
EDGX      0.548      0.513                0.546                  0.497
XNYS      0.321      0.319                0.300                  0.302
XCHI      0.394      0.349                0.281                  0.275
IEXG      0.308      0.296                0.250                  0.208
EPRL      0.611      0.534                  NaN                    NaN
XBOS      0.577      0.562                  NaN                    NaN
EDGA      0.487      0.496                  NaN                    NaN
XPSX      0.398      0.456                  NaN                    NaN
BATY      0.364      0.345                  NaN                    NaN
XASE      0.173      0.190                  NaN                    NaN
```

### Venue activity
```
venue  updates  share_of_all_updates  updates_per_s  updates_per_s_in_big_moves  median_spread_cents  median_spread_bps
 XNAS   276774                 0.295          4.805                      62.155                 19.0              3.076
 ARCX   154019                 0.164          2.674                      45.475                 31.0              4.973
 BATS   146250                 0.156          2.539                      29.380                 28.0              4.552
 EDGX    83988                 0.089          1.458                      29.235                 39.0              6.321
 XNYS    61125                 0.065          1.061                      13.170                 56.0              9.100
 IEXG    52755                 0.056          0.916                      12.415                147.0             23.861
 MEMX    41346                 0.044          0.718                      11.355                 50.0              8.077
 EPRL    23407                 0.025          0.406                       6.215                246.0             39.894
 XBOS    23159                 0.025          0.402                      11.400               1186.0            192.393
 XPSX    21424                 0.023          0.372                       7.480               3373.0            552.093
 EDGA    20159                 0.021          0.350                       6.660                204.0             33.012
 BATY    17742                 0.019          0.308                       6.485                729.0            118.382
 XCHI     9423                 0.010          0.164                       2.330                125.0             20.386
 XASE     7103                 0.008          0.123                       1.055               4433.0            742.341
```

## Reading these numbers
- Big moves are picked from the data and are retrospective; they are not tied to a news timestamp.
- 'First to halfway' and leadership count who moved first, not who caused the move.
- The NBBO is built only from the venues loaded here; it is not the official SIP NBBO.
- ts_recv includes each feed's path to Databento; ts_event uses each exchange's own clock.
- One day, one symbol: descriptive only, no confidence intervals.
- Outside 09:30-16:00 some venues leave old quotes standing, so the combined best bid/offer is
  often locked or crossed; read pre-market and after-hours inside-quote shares with care.
