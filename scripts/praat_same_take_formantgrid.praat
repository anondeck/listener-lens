form Same-take FormantGrid identity/shift resynthesis
    sentence Input_WAV
    sentence Output_WAV
    real Edit_start_s 0.0
    real Edit_end_s 0.1
    positive Output_sample_rate_hz 24000
    real Alpha 0.0
    real F1_delta_Bark 0.0
    real F2_delta_Bark 0.0
    positive Shift_taper_s 0.02
endform

original = Read from file: input_WAV$
selectObject: original
resampled = Resample: 11000, 50
selectObject: resampled
lpc = To LPC (burg): 10, 0.025, 0.005, 50
selectObject: lpc
formant = To Formant
selectObject: formant
grid = Down to FormantGrid

selectObject: formant
number_of_frames = Get number of frames
for iframe to number_of_frames
    selectObject: formant
    time = Get time from frame number: iframe
    if time >= edit_start_s and time <= edit_end_s
        envelope = 1
        if time < edit_start_s + shift_taper_s
            envelope = (time - edit_start_s) / shift_taper_s
        elsif time > edit_end_s - shift_taper_s
            envelope = (edit_end_s - time) / shift_taper_s
        endif
        envelope = max (0, min (1, envelope))

        selectObject: formant
        f1 = Get value at time: 1, time, "Hertz", "Linear"
        f2 = Get value at time: 2, time, "Hertz", "Linear"
        if f1 <> undefined
            f1_bark = 26.81 / (1 + 1960 / f1) - 0.53
            shifted_f1_bark = f1_bark + alpha * f1_delta_Bark * envelope
            shifted_f1 = 1960 / (26.81 / (shifted_f1_bark + 0.53) - 1)
            selectObject: grid
            Remove formant points between: 1, time - 0.000001, time + 0.000001
            Add formant point: 1, time, shifted_f1
        endif
        if f2 <> undefined
            f2_bark = 26.81 / (1 + 1960 / f2) - 0.53
            shifted_f2_bark = f2_bark + alpha * f2_delta_Bark * envelope
            shifted_f2 = 1960 / (26.81 / (shifted_f2_bark + 0.53) - 1)
            selectObject: grid
            Remove formant points between: 2, time - 0.000001, time + 0.000001
            Add formant point: 2, time, shifted_f2
        endif
    endif
endfor

selectObject: resampled, lpc
source = Filter (inverse)
selectObject: source, grid
filtered = Filter (no scale)
selectObject: filtered
output = Resample: output_sample_rate_hz, 50
selectObject: output
Save as WAV file: output_WAV$
