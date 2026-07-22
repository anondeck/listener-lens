form Sentence-pair-v2 Burg interval measurement
    sentence Input_WAV
    sentence Output_TSV
    real Interval_start_s
    real Interval_end_s
    real Maximum_formant_Hz
endform

sound = Read from file: input_WAV$
selectObject: sound
formant = To Formant (burg): 0.005, 5.0, maximum_formant_Hz, 0.025, 50

writeFileLine: output_TSV$, "time_s", tab$, "f1_hz", tab$, "f2_hz", tab$, "f3_hz", tab$, "f4_hz"
first_frame = ceiling (interval_start_s / 0.005)
last_frame = floor (interval_end_s / 0.005)
for iframe from first_frame to last_frame
    time = iframe * 0.005
    selectObject: formant
    f1 = Get value at time: 1, time, "Hertz", "Linear"
    f2 = Get value at time: 2, time, "Hertz", "Linear"
    f3 = Get value at time: 3, time, "Hertz", "Linear"
    f4 = Get value at time: 4, time, "Hertz", "Linear"
    appendFileLine: output_TSV$, fixed$ (time, 6), tab$, fixed$ (f1, 6), tab$, fixed$ (f2, 6), tab$, fixed$ (f3, 6), tab$, fixed$ (f4, 6)
endfor
