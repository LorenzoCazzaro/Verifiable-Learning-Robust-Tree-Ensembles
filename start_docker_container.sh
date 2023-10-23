#!/bin/bash

docker build -t vm .

docker run -it vm 