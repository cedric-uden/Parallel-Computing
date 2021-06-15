#pragma once

#include <chrono>
#include <iostream>
#include <sstream>

#define ERR_MSG "Error, "

enum class TimerState {
    TimerInitialized,
    TimerSetStart,
    TimerSetEnd
};

enum class TimerUnits {
    nanoseconds,
    microseconds,
    milliseconds
};


class Runtime_Analysis {

private:
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;

    std::string name;
    TimerState state;

    std::stringstream printDidSucceed(TimerUnits unit);

    std::stringstream printDidFail();

    std::stringstream callDurationCast(TimerUnits unit);

public:

    explicit Runtime_Analysis(const std::string name);

    void setStart();

    void setEnd();

    std::stringstream print(TimerUnits unit);

};
