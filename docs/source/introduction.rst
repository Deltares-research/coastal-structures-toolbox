.. highlight:: shell

============
Introduction
============


Purpose
=======

The Deltares Coastal Structures Toolbox package implements a large selection of (design) formulas for coastal structures. The intent is to make applying these formulas as easy as possible for the user, so most formulas are implemented in multiple ways to enable the calculation of different variables. When applying the formulas from the toolbox, the validity range of those formulas (where available) is checked and a warning is given when the formula is used outside its validity range (including which variable is out of range). The toolbox contains a significant number of tests to ensure the enduring quality of the implementations.

Structure
=========

The package is split into a hydraulic part, containing functions related to the hydraulic load on or hydraulic performance of a structure, and a structural part, concerning the structural performance. Withing these two parts, the modules are grouped together based on either physical mechanism or structural component. The resulting structure looks as follows:

* Hydraulic
    * Wave overtopping
    * Wave runup
    * Wave transmission
* Structural
    * Forces on caissons
    * Forces on crest walls
    * Stability of concrete armour units
    * Stability of rock armour
    * Stability of rock armour rear side
    * Stability of toe berm
