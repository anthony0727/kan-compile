Try speeding up Kolmogorov-Arnold Network (KAN) with `torch.compile`

* https://github.com/Blealtan/efficient-kan/tree/master
* https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#torchdynamo-and-fx-graphs

### train
```
eager train time 0: 0.20260166931152343
eager train time 1: 0.006864960193634033
eager train time 2: 0.0061287999153137205
eager train time 3: 0.006401023864746094
eager train time 4: 0.006948863983154297
eager train time 5: 0.007214079856872559
eager train time 6: 0.007699423789978027
eager train time 7: 0.008762368202209473
eager train time 8: 0.009631744384765625
eager train time 9: 0.009680895805358887
~~~~~~~~~~
********** cudagraphs **********
compile train time 0: 1.6030977783203124
compile train time 1: 0.39311154174804686
compile train time 2: 0.002600640058517456
compile train time 3: 0.0017489919662475586
compile train time 4: 0.0016641919612884522
compile train time 5: 0.0018472959995269776
compile train time 6: 0.0018657280206680299
compile train time 7: 0.0018595839738845825
compile train time 8: 0.0018350720405578614
compile train time 9: 0.0018965760469436645
~~~~~~~~~~
(train) eager median: 0.0074567518234252925, compile median: 0.0018626559972763062, speedup: 4.00328983683999x
~~~~~~~~~~
********** inductor **********
compile train time 0: 7.799271484375
compile train time 1: 0.003989824056625366
compile train time 2: 0.003055423974990845
compile train time 3: 0.0029685440063476564
compile train time 4: 0.0030318078994750978
compile train time 5: 0.0030412800312042236
compile train time 6: 0.0030915520191192626
compile train time 7: 0.003292095899581909
compile train time 8: 0.0036374399662017823
compile train time 9: 0.003347520112991333
~~~~~~~~~~
(train) eager median: 0.0074567518234252925, compile median: 0.003191823959350586, speedup: 2.3362039756548656x
~~~~~~~~~~
```

### eval
```
eager eval time 0: 0.14474342346191407
eager eval time 1: 0.001794111967086792
eager eval time 2: 0.0014336960315704346
eager eval time 3: 0.0013844480514526368
eager eval time 4: 0.001355936050415039
eager eval time 5: 0.0013557759523391724
eager eval time 6: 0.0013158400058746339
eager eval time 7: 0.0013700799942016602
eager eval time 8: 0.0013690880537033082
eager eval time 9: 0.0012974079847335816
~~~~~~~~~~
********** cudagraphs **********
compile eval time 0: 0.790523193359375
compile eval time 1: 0.007359488010406494
compile eval time 2: 0.00770854377746582
compile eval time 3: 0.0071998720169067386
compile eval time 4: 0.007142399787902832
compile eval time 5: 0.0071823358535766605
compile eval time 6: 0.007169919967651367
compile eval time 7: 0.007156576156616211
compile eval time 8: 0.007163904190063477
compile eval time 9: 0.007112703800201416
~~~~~~~~~~
(eval) eager median: 0.0012386720180511475, compile median: 0.007176127910614014, speedup: 0.17261008073992964x
~~~~~~~~~~
********** inductor **********
compile eval time 0: 3.697757080078125
compile eval time 1: 0.0005071679949760437
compile eval time 2: 0.0004362240135669708
compile eval time 3: 0.00044460800290107726
compile eval time 4: 0.0004411199986934662
compile eval time 5: 0.000434112012386322
compile eval time 6: 0.0004362240135669708
compile eval time 7: 0.0004280320107936859
compile eval time 8: 0.0004359360039234161
compile eval time 9: 0.000442656010389328
~~~~~~~~~~
(eval) eager median: 0.0012386720180511475, compile median: 0.0004386720061302185, speedup: 2.8236860359023033x
~~~~~~~~~~
********** tvm **********
compile eval time 0: 5.66759033203125
compile eval time 1: 0.0008355839848518371
compile eval time 2: 0.0005519359707832337
compile eval time 3: 0.0005097600221633911
compile eval time 4: 0.0005152639746665954
compile eval time 5: 0.0005160959959030152
compile eval time 6: 0.0005488640069961548
compile eval time 7: 0.0005591679811477661
compile eval time 8: 0.0005466880202293396
compile eval time 9: 0.0005294079780578613
~~~~~~~~~~
(eval) eager median: 0.0012386720180511475, compile median: 0.0005477760136127472, speedup: 2.2612746583804095x
~~~~~~~~~~
```