# META across venues, 2026-09-09 (rth session)
Venues: ARCX, BATS, BATY, EDGA, EDGX, EPRL, IEXG, MEMX, XASE, XBOS, XCHI, XNAS, XNYS, XPSX. Rows: 1,764,166.

## Clock: ts_recv
Big moves found: 20 (>= 10.0 bps over 10 s; first 5 min after the 09:30 open and around the 16:00 close skipped).

Midpoint measures use only quotes with spread <= 25.0 bps. Core (typically tight) venues: XNAS, ARCX, BATS, EDGX, XNYS, MEMX, XCHI.

### Biggest moves
```
 move      p0      p1  move_bps start_ny
    1 643.688 645.240    24.119 09:35:00
    2 645.300 642.020   -50.829 09:35:14
    3 642.470 644.350    29.262 09:36:50
    4 645.680 643.235   -37.867 09:37:47
    5 642.630 644.738    32.795 09:38:35
    6 644.440 646.072    25.332 09:39:09
    7 647.420 645.440   -30.583 09:39:53
    8 646.205 644.535   -25.843 09:41:49
    9 645.262 643.645   -25.067 09:42:13
   10 644.155 645.665    23.442 09:42:29
   11 645.452 647.085    25.292 09:44:32
   12 646.475 648.645    33.567 09:45:04
   13 648.025 646.258   -27.275 09:58:26
   14 647.380 645.525   -28.654 09:59:20
   15 645.035 642.768   -35.153 10:22:39
   16 641.660 643.295    25.481 10:28:38
   17 650.822 652.463    25.199 10:58:21
   18 652.688 654.395    26.161 10:58:39
   19 653.668 655.278    24.630 11:00:05
   20 656.030 653.560   -37.651 15:54:52
```

### Who reaches the new price first (big moves)
```
venue  n_moves  n_reached_halfway  first_share  median_rank  median_lag_behind_first_us  p90_lag_behind_first_us
 XNYS     20.0               20.0         0.35          2.5                     7610.53               1583402.10
 BATS     20.0               20.0         0.20          4.0                   194416.88               2129916.19
 EPRL     20.0               19.0         0.30          3.0                   222183.27               5947319.15
 EDGX     20.0               20.0         0.00          5.5                   322072.52               1754034.80
 XNAS     20.0               20.0         0.00          6.0                   494611.62               1905485.34
 IEXG     20.0               20.0         0.05          6.5                   613346.96               8063480.09
 ARCX     20.0               20.0         0.00          6.5                   649806.92               2189009.44
 MEMX     20.0               20.0         0.00          5.0                   941996.28               3380066.71
 EDGA     20.0               13.0         0.10          4.0                  1716289.56              11694021.55
 XCHI     20.0               20.0         0.00          9.5                  2289106.99               8184205.68
 BATY     20.0                6.0         0.00          9.0                  2798625.24               9429154.13
 XPSX     20.0               19.0         0.00         10.0                  2959627.09               9583096.71
 XBOS     20.0               11.0         0.00         11.0                  5278383.94              11711013.28
 XASE     20.0                8.0         0.00         11.5                  7587049.20              12620297.36
```

### Time at the inside quote (whole session vs big moves)
```
       pct_time_at_best_bid         pct_time_at_best_ask         avg_bid_behind_best_cents         
window            big_moves session            big_moves session                 big_moves  session
venue                                                                                              
ARCX                  42.00   39.59                38.34   53.25                      8.43     5.94
BATS                  25.05   36.51                29.21   33.13                     26.24    10.60
BATY                   3.15   11.08                 0.58    1.65                    674.29   280.98
EDGA                   0.07   13.59                 0.39    3.66                    431.27   143.73
EDGX                  19.75   26.60                14.77   21.66                     24.25    12.08
EPRL                   1.34    1.12                 1.11    0.86                    205.47   199.52
IEXG                   5.43    3.84                 3.38    6.96                    214.91   125.87
MEMX                   6.45    9.59                 4.93    6.87                     41.44    31.94
XASE                   1.77    1.87                 0.66    0.92                    337.71   588.92
XBOS                   0.81    3.26                 0.01    0.59                   2316.60  1002.12
XCHI                   0.08    0.64                 0.02    0.94                     72.96    66.36
XNAS                  68.30   74.46                68.01   70.86                      3.39     1.58
XNYS                   4.40    9.64                 5.29   11.01                     49.08    21.82
XPSX                   1.10    0.89                 0.17    0.34                   2731.14  3190.94
```

Combined NBBO locked or crossed: 0.014% of the session.

### Leadership share (reaction-latency engine)
```
        session  big_moves  session_core_venues  big_moves_core_venues
leader                                                                
MEMX      0.615      0.619                0.613                  0.615
BATS      0.609      0.612                0.605                  0.598
XNAS      0.609      0.583                0.584                  0.555
EDGX      0.535      0.547                0.522                  0.522
ARCX      0.477      0.508                0.423                  0.452
XNYS      0.313      0.326                0.239                  0.250
XCHI      0.443      0.280                0.242                  0.124
EPRL      0.716      0.715                  NaN                    NaN
XBOS      0.506      0.474                  NaN                    NaN
EDGA      0.443      0.417                  NaN                    NaN
BATY      0.358      0.262                  NaN                    NaN
XPSX      0.339      0.385                  NaN                    NaN
IEXG      0.273      0.303                  NaN                    NaN
XASE      0.206      0.189                  NaN                    NaN
```

### Venue activity
```
venue  updates  share_of_all_updates  updates_per_s  updates_per_s_in_big_moves  median_spread_cents  median_spread_bps
 XNAS   478498                 0.271         20.449                      65.635                 16.0              2.440
 ARCX   296340                 0.168         12.664                      47.345                 26.0              4.020
 BATS   256943                 0.146         10.980                      32.255                 26.0              3.980
 EDGX   164918                 0.093          7.048                      24.715                 32.0              4.906
 XNYS   137599                 0.078          5.880                      16.245                 52.0              7.979
 IEXG    91543                 0.052          3.912                      13.305                175.0             26.995
 MEMX    85602                 0.049          3.658                      13.960                 43.0              6.583
 EPRL    60601                 0.034          2.590                       9.140                178.0             27.405
 XPSX    51684                 0.029          2.209                       9.660               3417.0            539.917
 EDGA    43402                 0.025          1.855                       4.765                187.0             28.614
 XBOS    36373                 0.021          1.554                       6.340                766.0            117.905
 BATY    31874                 0.018          1.362                       3.125                279.0             42.805
 XASE    14836                 0.008          0.634                       1.410                467.0             72.107
 XCHI    13953                 0.008          0.596                       2.035                129.0             19.835
```

## Clock: ts_event
Big moves found: 20 (>= 10.0 bps over 10 s; first 5 min after the 09:30 open and around the 16:00 close skipped).

Midpoint measures use only quotes with spread <= 25.0 bps. Core (typically tight) venues: XNAS, ARCX, BATS, EDGX, XNYS, MEMX, XCHI.

### Biggest moves
```
 move      p0      p1  move_bps start_ny
    1 643.688 645.240    24.119 09:35:00
    2 645.300 642.020   -50.829 09:35:14
    3 642.470 644.350    29.262 09:36:50
    4 645.680 643.235   -37.867 09:37:47
    5 642.630 644.738    32.795 09:38:35
    6 644.440 646.072    25.332 09:39:09
    7 647.420 645.440   -30.583 09:39:53
    8 646.205 644.535   -25.843 09:41:49
    9 645.262 643.645   -25.067 09:42:13
   10 644.155 645.665    23.442 09:42:29
   11 645.452 647.085    25.292 09:44:32
   12 646.475 648.645    33.567 09:45:04
   13 648.025 646.258   -27.275 09:58:26
   14 647.380 645.525   -28.654 09:59:20
   15 645.035 642.768   -35.153 10:22:39
   16 641.660 643.295    25.481 10:28:38
   17 650.822 652.463    25.199 10:58:21
   18 652.688 654.395    26.161 10:58:39
   19 653.668 655.278    24.630 11:00:05
   20 656.030 653.560   -37.651 15:54:52
```

### Who reaches the new price first (big moves)
```
venue  n_moves  n_reached_halfway  first_share  median_rank  median_lag_behind_first_us  p90_lag_behind_first_us
 XNYS     20.0               20.0         0.35          2.5                     7426.95               1583227.70
 BATS     20.0               20.0         0.20          4.0                   194494.54               2130101.81
 EPRL     20.0               19.0         0.30          3.0                   222374.58               5947378.53
 EDGX     20.0               20.0         0.00          5.5                   322166.95               1754031.55
 XNAS     20.0               20.0         0.00          5.5                   494563.51               1905543.93
 IEXG     20.0               20.0         0.05          6.5                   613255.02               8063425.27
 ARCX     20.0               20.0         0.00          6.5                   649706.65               2188917.93
 MEMX     20.0               20.0         0.00          6.0                   941997.35               3380241.24
 EDGA     20.0               13.0         0.10          4.0                  1716295.30              11694170.38
 XCHI     20.0               20.0         0.00          9.5                  2289120.93               8184018.02
 BATY     20.0                6.0         0.00          9.0                  2798742.12               9429228.61
 XPSX     20.0               19.0         0.00         10.0                  2959495.84               9582955.61
 XBOS     20.0               11.0         0.00         11.0                  5278327.72              11711059.77
 XASE     20.0                8.0         0.00         11.5                  7586863.22              12620252.45
```

### Time at the inside quote (whole session vs big moves)
```
       pct_time_at_best_bid         pct_time_at_best_ask         avg_bid_behind_best_cents         
window            big_moves session            big_moves session                 big_moves  session
venue                                                                                              
ARCX                  42.00   39.59                38.34   53.25                      8.43     5.94
BATS                  25.04   36.50                29.20   33.13                     26.24    10.60
BATY                   3.15   11.08                 0.58    1.65                    674.29   280.98
EDGA                   0.07   13.59                 0.39    3.66                    431.27   143.73
EDGX                  19.75   26.60                14.78   21.66                     24.24    12.08
EPRL                   1.34    1.12                 1.11    0.86                    205.46   199.52
IEXG                   5.43    3.84                 3.38    6.96                    214.92   125.88
MEMX                   6.45    9.59                 4.93    6.87                     41.44    31.94
XASE                   1.76    1.87                 0.66    0.92                    337.71   588.92
XBOS                   0.81    3.26                 0.01    0.59                   2316.59  1002.12
XCHI                   0.08    0.64                 0.02    0.94                     72.95    66.36
XNAS                  68.31   74.47                68.02   70.86                      3.39     1.58
XNYS                   4.40    9.64                 5.29   11.01                     49.08    21.82
XPSX                   1.10    0.89                 0.18    0.34                   2731.14  3190.94
```

Combined NBBO locked or crossed: 0.014% of the session.

### Leadership share (reaction-latency engine)
```
        session  big_moves  session_core_venues  big_moves_core_venues
leader                                                                
XNAS      0.650      0.623                0.636                  0.611
BATS      0.581      0.573                0.572                  0.560
MEMX      0.531      0.537                0.519                  0.519
EDGX      0.538      0.539                0.527                  0.518
ARCX      0.507      0.547                0.458                  0.498
XNYS      0.335      0.355                0.260                  0.277
XCHI      0.458      0.279                0.258                  0.121
EPRL      0.677      0.675                  NaN                    NaN
XBOS      0.521      0.499                  NaN                    NaN
EDGA      0.443      0.408                  NaN                    NaN
XPSX      0.373      0.423                  NaN                    NaN
BATY      0.369      0.269                  NaN                    NaN
IEXG      0.285      0.314                  NaN                    NaN
XASE      0.228      0.223                  NaN                    NaN
```

### Venue activity
```
venue  updates  share_of_all_updates  updates_per_s  updates_per_s_in_big_moves  median_spread_cents  median_spread_bps
 XNAS   478498                 0.271         20.449                      65.635                 16.0              2.440
 ARCX   296340                 0.168         12.664                      47.345                 26.0              4.020
 BATS   256943                 0.146         10.980                      32.255                 26.0              3.980
 EDGX   164918                 0.093          7.048                      24.715                 32.0              4.906
 XNYS   137599                 0.078          5.880                      16.245                 52.0              7.979
 IEXG    91543                 0.052          3.912                      13.305                175.0             26.995
 MEMX    85602                 0.049          3.658                      13.960                 43.0              6.583
 EPRL    60601                 0.034          2.590                       9.140                178.0             27.405
 XPSX    51684                 0.029          2.209                       9.660               3417.0            539.917
 EDGA    43402                 0.025          1.855                       4.765                187.0             28.614
 XBOS    36373                 0.021          1.554                       6.340                766.0            117.905
 BATY    31874                 0.018          1.362                       3.125                279.0             42.805
 XASE    14836                 0.008          0.634                       1.410                467.0             72.107
 XCHI    13953                 0.008          0.596                       2.035                129.0             19.835
```

## Reading these numbers
- Big moves are picked from the data and are retrospective; they are not tied to a news timestamp.
- 'First to halfway' and leadership count who moved first, not who caused the move.
- The NBBO is built only from the venues loaded here; it is not the official SIP NBBO.
- ts_recv includes each feed's path to Databento; ts_event uses each exchange's own clock.
- One day, one symbol: descriptive only, no confidence intervals.
- Outside 09:30-16:00 some venues leave old quotes standing, so the combined best bid/offer is
  often locked or crossed; read pre-market and after-hours inside-quote shares with care.
