form Same-take FormantPath interval measurement
    sentence Input_WAV
    sentence Output_TSV
    real Interval_start_s
    real Interval_end_s
    real Maximum_number_of_formants
endform

sound = Read from file: input_WAV$
selectObject: sound
path = To FormantPath (burg): 0.005, maximum_number_of_formants, 5500, 0.025, 50, 0.05, 4
selectObject: path
table = Down to Table (optimal interval): interval_start_s, interval_end_s, "3 3 3 3", 1.25, "no", "yes", 6, "no", 3, "yes", 6, "yes", "yes", "yes"
selectObject: table
Save as tab-separated file: output_TSV$

