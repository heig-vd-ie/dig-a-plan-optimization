#!/bin/sh

DELAY_TIME=${1:-1} # Default delay time

# Each experiment: "description|command" (no leading blank line)
EXPERIMENTS="ieee33 fixed switches|python experiments/expansion_planning_script.py --kace ieee_33 --fixed_switches true --cachename ieee33_fixed_switches
ieee33 expectation|python experiments/expansion_planning_script.py --kace ieee_33 --cachename ieee33_expectation
ieee33 wasserstein|python experiments/expansion_planning_script.py --kace ieee_33 --riskmeasuretype Wasserstein --cachename ieee33_wasserstein
ieee33 worstcase|python experiments/expansion_planning_script.py --kace ieee_33 --riskmeasuretype WorstCase --cachename ieee33_worstcase
boisy feeder_1 fixed switches|python experiments/expansion_planning_script.py --kace boisy --feedername feeder_1 --fixed_switches true --cachename boisy_feeder_1_fixed_switches
boisy feeder_2 fixed switches|python experiments/expansion_planning_script.py --kace boisy --feedername feeder_2 --fixed_switches true --cachename boisy_feeder_2_fixed_switches
boisy feeder_1 expectation|python experiments/expansion_planning_script.py --kace boisy --feedername feeder_1 --admmiter 3 --cachename boisy_feeder_1_expectation
boisy feeder_2 expectation|python experiments/expansion_planning_script.py --kace boisy --feedername feeder_2 --admmiter 3 --cachename boisy_feeder_2_expectation
estavayer centre_ville fixed switches|python experiments/expansion_planning_script.py --kace estavayer --feedername centre_ville --fixed_switches true --cachename estavayer_centre_ville_fixed_switches
estavayer centre_ville expectation|python experiments/expansion_planning_script.py --kace estavayer --feedername centre_ville --admmiter 3 --cachename estavayer_centre_ville_expectation
estavayer aumont fixed switches|python experiments/expansion_planning_script.py --kace estavayer --feedername aumont --fixed_switches true --cachename estavayer_aumont_fixed_switches
estavayer autoroutes fixed switches|python experiments/expansion_planning_script.py --kace estavayer --feedername autoroutes --fixed_switches true --cachename estavayer_autoroutes_fixed_switches
estavayer bel-air fixed switches|python experiments/expansion_planning_script.py --kace estavayer --feedername bel-air --fixed_switches true --cachename estavayer_bel-air_fixed_switches
estavayer tout_vent fixed switches|python experiments/expansion_planning_script.py --kace estavayer --feedername tout_vent --fixed_switches true --cachename estavayer_tout_vent_fixed_switches
estavayer zone_industrielle fixed switches|python experiments/expansion_planning_script.py --kace estavayer --feedername zone_industrielle --fixed_switches true --cachename estavayer_zone_industrielle_fixed_switches
estavayer st-aubin fixed switches|python experiments/expansion_planning_script.py --kace estavayer --feedername st-aubin --fixed_switches true --cachename estavayer_st-aubin_fixed_switches
estavayer aumont expectation|python experiments/expansion_planning_script.py --kace estavayer --feedername aumont --admmiter 3 --cachename estavayer_aumont_expectation
estavayer autoroutes expectation|python experiments/expansion_planning_script.py --kace estavayer --feedername autoroutes --admmiter 3 --cachename estavayer_autoroutes_expectation
estavayer bel-air expectation|python experiments/expansion_planning_script.py --kace estavayer --feedername bel-air --admmiter 3 --cachename estavayer_bel-air_expectation
estavayer tout_vent expectation|python experiments/expansion_planning_script.py --kace estavayer --feedername tout_vent --admmiter 3 --cachename estavayer_tout_vent_expectation
estavayer zone_industrielle expectation|python experiments/expansion_planning_script.py --kace estavayer --feedername zone_industrielle --admmiter 3 --cachename estavayer_zone_industrielle_expectation
estavayer st-aubin expectation|python experiments/expansion_planning_script.py --kace estavayer --feedername st-aubin --admmiter 3 --cachename estavayer_st-aubin_expectation"

# Show menu
i=0
echo "Available experiments:"
echo "$EXPERIMENTS" | while IFS= read -r line; do
    [ -z "$line" ] && continue
    desc=$(echo "$line" | cut -d'|' -f1)
    echo "$i: $desc"
    i=$((i + 1))
done

# Ask user for selection
echo "Enter indices of experiments to run (space-separated), or 'all':"
read selection

# Run selected experiments
if [ "$selection" = "all" ]; then
    echo "$EXPERIMENTS" | while IFS= read -r line; do
        [ -z "$line" ] && continue
        desc=$(echo "$line" | cut -d'|' -f1)
        cmd=$(echo "$line" | cut -d'|' -f2-)
        echo "Running: $desc"
        eval $cmd
        echo "Finished: $desc"
        sleep $DELAY_TIME
    done
else
    for idx in $selection; do
        line=$(echo "$EXPERIMENTS" | sed -n "$((idx + 1))p")
        [ -z "$line" ] && continue
        desc=$(echo "$line" | cut -d'|' -f1)
        cmd=$(echo "$line" | cut -d'|' -f2-)
        echo "Running: $desc"
        eval $cmd
        echo "Finished: $desc"
        sleep $DELAY_TIME
    done
fi