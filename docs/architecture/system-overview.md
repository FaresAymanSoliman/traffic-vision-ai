# TrafficVision AI System Architecture

## 1. Document Purpose

This document describes the initial software architecture of TrafficVision AI.

The architecture is designed for a solo-developed full-stack computer vision
application. It prioritizes clear component boundaries, incremental development,
testability, and a simple deployment process.

The first complete version processes uploaded MP4 traffic videos. Support for
webcams and RTSP cameras will be introduced only after the uploaded-video
workflow is stable.

## 2. System Overview

TrafficVision AI processes traffic video and generates information about:

- Detected vehicles
- Vehicle classes
- Vehicle trajectories
- Vehicle counts
- Estimated speeds
- Traffic density
- Potential traffic violations
- Processing performance

The results are stored by the backend and presented through a React dashboard.

The major processing pipeline is:

```text
Traffic video
    |
    v
Video ingestion
    |
    v
Vehicle detection
    |
    v
Vehicle tracking
    |
    v
Vehicle counting
    |
    v
Speed estimation
    |
    v
Traffic density estimation
    |
    v
Violation detection
    |
    v
Database and media storage
    |
    v
FastAPI endpoints
    |
    v
React dashboard