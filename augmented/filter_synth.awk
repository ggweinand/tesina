#!/bin/awk -f
BEGIN {
    FS=",";
    n_synth=50;
}
{
    if ($5 == "True") {
        if (seen[$1]++ < n_synth){
            print;
        }
    } else {
        print;
    }
}
