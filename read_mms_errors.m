Text = fileread('mms_output.txt');

ErrorStrings = regexp(Text, '(?<=Error = )[.0-9]{13}(?=\s)', 'match')