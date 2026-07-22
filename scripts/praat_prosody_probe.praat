form Controlled prosody probe
    sentence Input_WAV
    sentence Output_TSV
endform

sound = Read from file: input_WAV$
selectObject: sound
duration = Get total duration
pitch = To Pitch: 0.005, 75, 500

writeFileLine: output_TSV$, "time_s", tab$, "pitch_hz", tab$, "rms"
number_of_frames = floor (duration / 0.005)
for iframe to number_of_frames
    time = (iframe - 0.5) * 0.005
    selectObject: pitch
    pitch_hz = Get value at time: time, "Hertz", "Linear"
    left = max (0, time - 0.005)
    right = min (duration, time + 0.005)
    selectObject: sound
    rms = Get root-mean-square: left, right
    appendFileLine: output_TSV$, fixed$ (time, 6), tab$, fixed$ (pitch_hz, 6), tab$, fixed$ (rms, 12)
endfor
