#!/usr/bin/env bash
docker-compose stop
docker-compose build runner
sudo rm -rf ./experience/*
docker-compose up -d --force-recreate