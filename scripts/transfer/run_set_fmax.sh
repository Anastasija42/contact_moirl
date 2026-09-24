#!/usr/bin/env bash
# run_set_fmax.sh <stroke> <tag> <weights> : all seven bodies, per-body capacity -> <tag>/<body>/, progress in <tag>/bodies.done
mkdir -p $2; : > $2/bodies.done
for sp in human chimp bonobo australopithecus_prometheus australopithecus_sediba homo_naledi homo_neanderthal; do
  ./run_one_fmax.sh $1 $sp $2 $3; echo "$sp $(tail -n 1 $2/$sp/run.log)" >> $2/bodies.done
done
echo "ALL BODIES DONE $(date +%T)" >> $2/bodies.done
