form Same-take vowel-core probe
    sentence Input_WAV
    sentence Output_TSV
endform

sound = Read from file: input_WAV$
selectObject: sound
duration = Get total duration
pitch = To Pitch: 0.005, 75, 500
selectObject: sound
formant = To Formant (burg): 0.005, 5.0, 5500, 0.025, 50

writeFileLine: output_TSV$, "time_s", tab$, "f1_hz", tab$, "f2_hz", tab$, "f3_hz", tab$, "f4_hz", tab$, "pitch_hz", tab$, "rms"
number_of_frames = floor (duration / 0.005)
for iframe to number_of_frames
    time = (iframe - 0.5) * 0.005
    selectObject: formant
    f1 = Get value at time: 1, time, "Hertz", "Linear"
    f2 = Get value at time: 2, time, "Hertz", "Linear"
    f3 = Get value at time: 3, time, "Hertz", "Linear"
    f4 = Get value at time: 4, time, "Hertz", "Linear"
    selectObject: pitch
    pitch_hz = Get value at time: time, "Hertz", "Linear"
    left = max (0, time - 0.005)
    right = min (duration, time + 0.005)
    selectObject: sound
    rms = Get root-mean-square: left, right
    appendFileLine: output_TSV$, fixed$ (time, 6), tab$, fixed$ (f1, 6), tab$, fixed$ (f2, 6), tab$, fixed$ (f3, 6), tab$, fixed$ (f4, 6), tab$, fixed$ (pitch_hz, 6), tab$, fixed$ (rms, 12)
endfor
