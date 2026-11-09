# META across venues, 2026-09-09 (extended session)
Venues: ARCX, BATS, BATY, EDGA, EDGX, EPRL, IEXG, MEMX, XASE, XBOS, XCHI, XNAS, XNYS, XPSX. Rows: 2,001,276.

## Clock: ts_recv
Big moves found: 20 (>= 10.0 bps over 10 s; first 5 min after the 09:30 open and around the 16:00 close skipped).

Midpoint measures use only quotes with spread <= 25.0 bps. Core (typically tight) venues: XNAS, ARCX, BATS, EDGX, XNYS, MEMX, XCHI.

### Biggest moves
```
 move      p0      p1  move_bps start_ny
    1 621.750 623.820    33.293 04:52:59
    2 624.335 627.525    51.094 06:01:57
    3 627.980 630.920    46.817 06:05:40
    4 635.668 637.870    34.649 06:21:54
    5 632.855 635.320    38.950 07:01:44
    6 637.565 639.360    28.154 07:03:14
    7 643.478 645.235    27.313 09:06:01
    8 645.802 647.535    26.827 09:24:54
    9 648.480 650.540    31.767 09:27:49
   10 645.300 642.020   -50.829 09:35:14
   11 642.470 644.350    29.262 09:36:50
   12 645.680 643.235   -37.867 09:37:47
   13 642.630 644.738    32.795 09:38:35
   14 647.420 645.440   -30.583 09:39:53
   15 646.475 648.645    33.567 09:45:04
   16 648.025 646.258   -27.275 09:58:26
   17 647.380 645.525   -28.654 09:59:20
   18 645.035 642.768   -35.153 10:22:39
   19 656.030 653.560   -37.651 15:54:52
   20 651.325 649.395   -29.632 17:29:46
```

### Who reaches the new price first (big moves)
```
venue  n_moves  n_reached_halfway  first_share  median_rank  median_lag_behind_first_us  p90_lag_behind_first_us
 XNYS     20.0               11.0         0.25          2.0                      408.11                198439.16
 EDGX     20.0               20.0         0.05          3.0                     1319.65               1346049.69
 XNAS     20.0               20.0         0.20          3.0                     7428.44               1674806.34
 ARCX     20.0               20.0         0.15          5.0                     9295.76               1847681.36
 BATS     20.0               19.0         0.10          4.0                    67529.83               2142162.46
 EDGA     20.0                7.0         0.10          4.0                   222209.77              10422687.00
 EPRL     20.0                9.0         0.05          4.0                   225718.49               4358708.99
 MEMX     20.0               15.0         0.10          5.0                   318751.89               3226248.42
 XBOS     20.0                7.0         0.00          9.0                   342039.60               7122699.38
 IEXG     20.0               10.0         0.00         10.0                  2222599.97               9190920.17
 XCHI     20.0               10.0         0.00         10.0                  2289106.99               9936693.55
 BATY     20.0                5.0         0.00          7.0                  2982827.65              10408621.24
 XPSX     20.0               12.0         0.00          9.5                  3668401.71              12076809.31
 XASE     20.0                5.0         0.00          9.0                  7038888.34              11909020.45
```

### Time at the inside quote (whole session vs big moves)
```
       pct_time_at_best_bid         pct_time_at_best_ask         avg_bid_behind_best_cents         
window            big_moves session            big_moves session                 big_moves  session
venue                                                                                              
ARCX                  31.06   30.73                37.59   51.30                     35.27    75.66
BATS                  16.58   16.68                19.07   18.42                    350.03   209.44
BATY                   9.62    6.72                 0.72    1.49                   1927.95  1695.21
EDGA                   0.02   10.72                 1.19    2.97                   1140.06   927.49
EDGX                  15.11   19.44                17.75   20.73                     49.22    84.06
EPRL                   0.19    0.51                 0.58    0.64                    954.14   927.85
IEXG                   5.00    3.84                 4.82    6.96                    267.69   126.94
MEMX                   3.08    7.38                 5.45    7.85                    148.80   152.11
XASE                   3.33    1.87                 0.01    0.92                    293.31   588.92
XBOS                   0.57    3.25                 0.01    0.59                   1841.47  1000.48
XCHI                   0.15    0.43                 0.01    0.75                    239.37   233.94
XNAS                  70.12   49.89                68.91   55.37                     19.12    73.48
XNYS                  10.25   43.09                 5.34    6.68                     97.64    22.45
XPSX                   0.42    0.88                 0.31    0.33                   2901.92  3211.65
```

Combined NBBO locked or crossed: 24.750% of the session.

### Leadership share (reaction-latency engine)
```
        session  big_moves  session_core_venues  big_moves_core_venues
leader                                                                
MEMX      0.615      0.615                0.611                  0.608
BATS      0.607      0.597                0.601                  0.576
XNAS      0.605      0.579                0.578                  0.544
EDGX      0.538      0.559                0.523                  0.525
ARCX      0.481      0.516                0.429                  0.463
XNYS      0.315      0.338                0.241                  0.256
XCHI      0.443      0.352                0.242                  0.150
EPRL      0.715      0.712                  NaN                    NaN
XBOS      0.497      0.474                  NaN                    NaN
EDGA      0.441      0.372                  NaN                    NaN
BATY      0.356      0.244                  NaN                    NaN
XPSX      0.338      0.411                  NaN                    NaN
IEXG      0.273      0.309                  NaN                    NaN
XASE      0.206      0.152                  NaN                    NaN
```

### Venue activity
```
venue  updates  share_of_all_updates  updates_per_s  updates_per_s_in_big_moves  median_spread_cents  median_spread_bps
 XNAS   550452                 0.275          9.556                      49.720                 17.0              2.614
 ARCX   365028                 0.182          6.337                      38.240                 28.0              4.305
 BATS   275273                 0.138          4.779                      21.195                 27.0              4.180
 EDGX   203542                 0.102          3.534                      22.870                 36.0              5.547
 XNYS   144277                 0.072          2.505                       9.790                 52.0              8.023
 MEMX    96994                 0.048          1.684                       9.365                 46.0              7.069
 IEXG    91639                 0.046          1.591                       7.660                175.0             26.995
 EPRL    61086                 0.031          1.061                       5.165                179.0             27.586
 XPSX    54788                 0.027          0.951                       5.735               3414.0            539.456
 EDGA    46772                 0.023          0.812                       3.735                188.0             28.755
 XBOS    45283                 0.023          0.786                       5.355                744.0            114.657
 BATY    35650                 0.018          0.619                       2.690                313.0             47.966
 XCHI    15630                 0.008          0.271                       1.590                149.0             22.809
 XASE    14862                 0.007          0.258                       1.010                467.0             72.107
```

## Clock: ts_event
Big moves found: 20 (>= 10.0 bps over 10 s; first 5 min after the 09:30 open and around the 16:00 close skipped).

Midpoint measures use only quotes with spread <= 25.0 bps. Core (typically tight) venues: XNAS, ARCX, BATS, EDGX, XNYS, MEMX, XCHI.

### Biggest moves
```
 move      p0      p1  move_bps start_ny
    1 621.750 623.820    33.293 04:52:59
    2 624.335 627.525    51.094 06:01:57
    3 627.980 630.920    46.817 06:05:40
    4 635.668 637.870    34.649 06:21:54
    5 632.855 635.320    38.950 07:01:44
    6 637.565 639.360    28.154 07:03:14
    7 643.478 645.235    27.313 09:06:01
    8 645.802 647.535    26.827 09:24:54
    9 648.480 650.540    31.767 09:27:49
   10 645.300 642.020   -50.829 09:35:14
   11 642.470 644.350    29.262 09:36:50
   12 645.680 643.235   -37.867 09:37:47
   13 642.630 644.738    32.795 09:38:35
   14 647.420 645.440   -30.583 09:39:53
   15 646.475 648.645    33.567 09:45:04
   16 648.025 646.258   -27.275 09:58:26
   17 647.380 645.525   -28.654 09:59:20
   18 645.035 642.768   -35.153 10:22:39
   19 656.030 653.560   -37.651 15:54:52
   20 651.325 649.395   -29.632 17:29:46
```

### Who reaches the new price first (big moves)
```
venue  n_moves  n_reached_halfway  first_share  median_rank  median_lag_behind_first_us  p90_lag_behind_first_us
 XNYS     20.0               11.0         0.25          2.0                      235.05                198242.83
 EDGX     20.0               20.0         0.05          3.0                     1504.56               1346064.91
 XNAS     20.0               20.0         0.20          2.5                     7584.75               1674675.63
 ARCX     20.0               20.0         0.15          4.5                     9108.25               1847515.22
 BATS     20.0               19.0         0.15          4.0                    67718.95               2142349.76
 EDGA     20.0                7.0         0.10          4.0                   222380.52              10422847.92
 EPRL     20.0                9.0         0.05          4.0                   225737.93               4358903.32
 MEMX     20.0               15.0         0.05          5.0                   318779.46               3226427.93
 XBOS     20.0                7.0         0.00          9.0                   341897.44               7122644.14
 IEXG     20.0               10.0         0.00         10.0                  2222685.21               9190692.72
 XCHI     20.0               10.0         0.00         10.0                  2289120.93               9936530.63
 BATY     20.0                5.0         0.00          7.0                  2982967.51              10408785.61
 XPSX     20.0               12.0         0.00          9.5                  3668361.46              12076674.50
 XASE     20.0                5.0         0.00          9.0                  7038710.35              11908938.33
```

### Time at the inside quote (whole session vs big moves)
```
       pct_time_at_best_bid         pct_time_at_best_ask         avg_bid_behind_best_cents         
window            big_moves session            big_moves session                 big_moves  session
venue                                                                                              
ARCX                  31.05   30.73                37.58   51.30                     35.27    75.66
BATS                  16.58   16.68                19.06   18.42                    350.03   209.44
BATY                   9.62    6.72                 0.72    1.49                   1927.94  1695.21
EDGA                   0.02   10.72                 1.19    2.97                   1140.05   927.49
EDGX                  15.11   19.44                17.75   20.73                     49.22    84.06
EPRL                   0.19    0.51                 0.57    0.64                    954.13   927.85
IEXG                   5.00    3.84                 4.82    6.95                    267.70   126.94
MEMX                   3.08    7.38                 5.45    7.85                    148.79   152.11
XASE                   3.32    1.87                 0.01    0.92                    293.31   588.92
XBOS                   0.57    3.25                 0.01    0.59                   1841.46  1000.48
XCHI                   0.15    0.43                 0.01    0.75                    239.37   233.94
XNAS                  70.12   49.89                68.91   55.37                     19.12    73.48
XNYS                  10.25   43.09                 5.34    6.68                     97.64    22.45
XPSX                   0.42    0.88                 0.31    0.33                   2901.92  3211.64
```

Combined NBBO locked or crossed: 24.750% of the session.

### Leadership share (reaction-latency engine)
```
        session  big_moves  session_core_venues  big_moves_core_venues
leader                                                                
XNAS      0.645      0.615                0.629                  0.594
BATS      0.577      0.559                0.567                  0.538
ARCX      0.514      0.566                0.468                  0.522
MEMX      0.530      0.532                0.516                  0.508
EDGX      0.539      0.542                0.525                  0.508
XNYS      0.338      0.370                0.264                  0.289
XCHI      0.458      0.356                0.257                  0.152
EPRL      0.676      0.679                  NaN                    NaN
XBOS      0.514      0.502                  NaN                    NaN
EDGA      0.440      0.358                  NaN                    NaN
XPSX      0.373      0.444                  NaN                    NaN
BATY      0.365      0.243                  NaN                    NaN
IEXG      0.285      0.321                  NaN                    NaN
XASE      0.228      0.189                  NaN                    NaN
```

### Venue activity
```
venue  updates  share_of_all_updates  updates_per_s  updates_per_s_in_big_moves  median_spread_cents  median_spread_bps
 XNAS   550452                 0.275          9.556                      49.720                 17.0              2.614
 ARCX   365028                 0.182          6.337                      38.240                 28.0              4.305
 BATS   275273                 0.138          4.779                      21.195                 27.0              4.180
 EDGX   203542                 0.102          3.534                      22.870                 36.0              5.547
 XNYS   144277                 0.072          2.505                       9.790                 52.0              8.023
 MEMX    96994                 0.048          1.684                       9.365                 46.0              7.069
 IEXG    91639                 0.046          1.591                       7.660                175.0             26.995
 EPRL    61086                 0.031          1.061                       5.165                179.0             27.586
 XPSX    54788                 0.027          0.951                       5.735               3414.0            539.456
 EDGA    46772                 0.023          0.812                       3.735                188.0             28.755
 XBOS    45283                 0.023          0.786                       5.355                744.0            114.657
 BATY    35650                 0.018          0.619                       2.690                313.0             47.966
 XCHI    15630                 0.008          0.271                       1.590                149.0             22.809
 XASE    14862                 0.007          0.258                       1.010                467.0             72.107
```

## Reading these numbers
- Big moves are picked from the data and are retrospective; they are not tied to a news timestamp.
- 'First to halfway' and leadership count who moved first, not who caused the move.
- The NBBO is built only from the venues loaded here; it is not the official SIP NBBO.
- ts_recv includes each feed's path to Databento; ts_event uses each exchange's own clock.
- One day, one symbol: descriptive only, no confidence intervals.
- Outside 09:30-16:00 some venues leave old quotes standing, so the combined best bid/offer is
  often locked or crossed; read pre-market and after-hours inside-quote shares with care.
