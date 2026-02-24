#!/bin/sh

# Set this variable to true to run only the first four experiments
BREAK_POINT=${1:-3} # Default is 3, which means run all experiments
DELAY_TIME=${2:-10} # Default delay time of 60 seconds between experiments

echo "Start Run with BREAK_POINT=$BREAK_POINT and DELAY_TIME=$DELAY_TIME seconds between experiments."

# First experiment
python experiments/expansion_planning_script.py --kace ieee_33 --fixed_switches true --cachename ieee33_fixed_switches
echo "Finished ieee33 fixed switches"
sleep $DELAY_TIME
# Second experiment
python experiments/expansion_planning_script.py --kace ieee_33 --cachename ieee33_expectation
echo "Finished ieee33_expectation"
sleep $DELAY_TIME
# Third experiment
python experiments/expansion_planning_script.py --kace ieee_33 --riskmeasuretype Wasserstein --cachename ieee33_wasserstein
echo "Finished ieee33 wasserstein"
sleep $DELAY_TIME
# Fourth experiment
python experiments/expansion_planning_script.py --kace ieee_33 --riskmeasuretype WorstCase --cachename ieee33_worstcase
echo "Finished ieee33 worstcase"
sleep $DELAY_TIME
# If BREAK_POINT is 1, exit here
if [ "$BREAK_POINT" = 1 ]; then
    echo "stopping after first four experiments."
    exit 0
fi

# Additional experiments
python experiments/expansion_planning_script.py --kace boisy --feedername feeder_1 --fixed_switches true --cachename boisy_feeder_1_fixed_switches
echo "Finished feeder_1_fixed_switches"
sleep $DELAY_TIME
python experiments/expansion_planning_script.py --kace boisy --feedername feeder_2 --fixed_switches true --cachename boisy_feeder_2_fixed_switches
echo "Finished feeder_2_fixed_switches"
sleep $DELAY_TIME

python experiments/expansion_planning_script.py --kace boisy --feedername feeder_1 --admmiter 3 --cachename boisy_feeder_1_expectation
echo "Finished feeder_1_expectation"
sleep $DELAY_TIME
python experiments/expansion_planning_script.py --kace boisy --feedername feeder_2 --admmiter 3 --cachename boisy_feeder_2_expectation
echo "Finished feeder_2_expectation"
sleep $DELAY_TIME

if [ "$BREAK_POINT" = 2 ]; then
    echo "stopping after first six experiments."
    exit 0
fi


python experiments/expansion_planning_script.py --kace estavayer --feedername centre_ville --fixed_switches true --cachename estavayer_centre_ville_fixed_switches
echo "Finished estavayer_centre_ville_fixed_switches"
sleep $DELAY_TIME
python experiments/expansion_planning_script.py --kace estavayer --feedername centre_ville --admmiter 3 --cachename estavayer_centre_ville_expectation
echo "Finished estavayer_centre_ville_expectation"
sleep $DELAY_TIME

if [ "$BREAK_POINT" = 3 ]; then
    echo "stopping after first seven experiments."
    exit 0
fi

python experiments/expansion_planning_script.py --kace estavayer --feedername aumont --fixed_switches true --cachename estavayer_aumont_fixed_switches
echo "Finished aumont"
sleep $DELAY_TIME
python experiments/expansion_planning_script.py --kace estavayer --feedername autoroutes --fixed_switches true --cachename estavayer_autoroutes_fixed_switches
echo "Finished autoroutes"
sleep $DELAY_TIME
python experiments/expansion_planning_script.py --kace estavayer --feedername bel-air --fixed_switches true --cachename estavayer_bel-air_fixed_switches
echo "Finished bel-air"
sleep $DELAY_TIME
python experiments/expansion_planning_script.py --kace estavayer --feedername tout_vent --fixed_switches true --cachename estavayer_tout_vent_fixed_switches
echo "Finished tout_vent"
sleep $DELAY_TIME
python experiments/expansion_planning_script.py --kace estavayer --feedername zone_industrielle --fixed_switches true --cachename estavayer_zone_industrielle_fixed_switches
echo "Finished zone_industrielle"
python experiments/expansion_planning_script.py --kace estavayer --feedername st-aubin --fixed_switches true --cachename estavayer_st-aubin_fixed_switches
echo "Finished st-aubin"
sleep $DELAY_TIME

python experiments/expansion_planning_script.py --kace estavayer --feedername aumont --admmiter 3 --cachename estavayer_aumont_expectation
echo "Finished aumont expectation"
sleep $DELAY_TIME
python experiments/expansion_planning_script.py --kace estavayer --feedername autoroutes --admmiter 3 --cachename estavayer_autoroutes_expectation
echo "Finished autoroutes expectation"
sleep $DELAY_TIME
python experiments/expansion_planning_script.py --kace estavayer --feedername bel-air --admmiter 3 --cachename estavayer_bel-air_expectation
echo "Finished bel-air expectation"
sleep $DELAY_TIME
python experiments/expansion_planning_script.py --kace estavayer --feedername tout_vent --admmiter 3 --cachename estavayer_tout_vent_expectation
echo "Finished tout_vent expectation"
sleep $DELAY_TIME
python experiments/expansion_planning_script.py --kace estavayer --feedername zone_industrielle --admmiter 3 --cachename estavayer_zone_industrielle_expectation
echo "Finished zone_industrielle expectation"
sleep $DELAY_TIME
python experiments/expansion_planning_script.py --kace estavayer --feedername st-aubin --admmiter 3 --cachename estavayer_st-aubin_expectation
echo "Finished st-aubin expectation"    
