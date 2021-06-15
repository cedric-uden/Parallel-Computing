#include "Runtime_Analysis.h"

Runtime_Analysis::Runtime_Analysis(const std::string name) {
    this->name = name;
    this->state = TimerState::TimerInitialized;
}


void Runtime_Analysis::setStart() {
    start = std::chrono::steady_clock::now();
    this->state = TimerState::TimerSetStart;
}

void Runtime_Analysis::setEnd() {
    if (this->state == TimerState::TimerSetStart) {
        end = std::chrono::steady_clock::now();
        this->state = TimerState::TimerSetEnd;
    } else if (this->state == TimerState::TimerSetEnd) {
        end = std::chrono::steady_clock::now();
    } else {
        std::cerr << "Use start first!" << std::endl;
    }
}


std::stringstream Runtime_Analysis::print(TimerUnits unit) {

    if (this->state == TimerState::TimerSetEnd) {
        return printDidSucceed(unit);
    } else {
        return printDidFail();
    }
}

std::stringstream Runtime_Analysis::printDidFail() {
    std::stringstream out;
    out << ERR_MSG << " start and / or End point are not set" << std::endl;
    return out;
}

std::stringstream Runtime_Analysis::printDidSucceed(TimerUnits unit) {

    std::stringstream out;
    std::string printUnit = "";

    out << name << ": " << "Time difference = ";

    if (unit == TimerUnits::nanoseconds) {
        out << callDurationCast(unit).rdbuf();
        printUnit = "nS";
    } else if (unit == TimerUnits::microseconds) {
        out << callDurationCast(unit).rdbuf();
        printUnit = "µS";
    } else if (unit == TimerUnits::milliseconds) {
        out << callDurationCast(unit).rdbuf();
        printUnit = "mS";
    }
    out << "[" << printUnit << "]" << std::endl;

    return out;
}


std::stringstream Runtime_Analysis::callDurationCast(TimerUnits unit) {

    std::stringstream out;
    if (unit == TimerUnits::nanoseconds) {
        out << std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - start).count();
    } else if (unit == TimerUnits::microseconds) {
        out << std::chrono::duration_cast<std::chrono::microseconds>(
                end - start).count();
    } else if (unit == TimerUnits::milliseconds) {
        out << std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start).count();
    }
    return out;

}
