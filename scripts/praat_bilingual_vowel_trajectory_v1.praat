form Bilingual vowel trajectory measurement v1
    sentence Input_WAV
    sentence Output_TSV
    real Maximum_formant_Hz
endform

sound = Read from file: input_WAV$
selectObject: sound
formant = To Formant (burg): 0.005, 5.0, maximum_formant_Hz, 0.025, 50

writeFileLine: output_TSV$, "time_s", tab$, "f1_hz", tab$, "f2_hz", tab$, "f3_hz"
selectObject: formant
frame_count = Get number of frames
for iframe from 1 to frame_count
    time = Get time from frame number: iframe
    f1 = Get value at time: 1, time, "Hertz", "Linear"
    f2 = Get value at time: 2, time, "Hertz", "Linear"
    f3 = Get value at time: 3, time, "Hertz", "Linear"
    appendFileLine: output_TSV$, fixed$ (time, 6), tab$, fixed$ (f1, 6), tab$, fixed$ (f2, 6), tab$, fixed$ (f3, 6)
endfor
