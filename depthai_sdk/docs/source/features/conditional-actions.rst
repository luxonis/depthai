Conditional actions
===================

DepthAI SDK provides a way to perform actions based on some conditions.
For example, you can perform an action when a certain number of objects is detected in the frame.
This functionality can be achieved by using Trigger-Action API.

Overview
--------
Trigger-Action API is a way to define a set of conditions and actions that should be performed when these conditions are met.
DepthAI SDK provides a set of predefined conditions and actions, but you can also define your own.

Basic concepts:

- **Trigger** - a condition that should be met to perform an action.
- **Action** - an action that should be performed when a trigger is met.

.. note:: Trigger-Action API is implemented in the :mod:`depthai.trigger_action` module.

Triggers
--------

The base class for all triggers is :class:`Trigger <depthai_sdk.trigger_action.Trigger>`.
In order to create a trigger, you need to use the :class:`Trigger <depthai_sdk.trigger_action.Trigger>` class and pass the following parameters:

- ``input`` - a component that should be used as a trigger source.
- ``condition`` - a function that should return ``True`` or ``False`` based on the trigger source.
- ``cooldown`` - defines how often a trigger can be activated (in seconds).

The set of predefined triggers:

- :class:`DetectionTrigger <depthai_sdk.trigger_action.DetectionTrigger>` - a trigger that is activated when a certain number of objects is detected in the frame.

Actions
-------

An action can be represented by either a function or a class derived from :class:`Action <depthai_sdk.trigger_action.Action>` class.
The custom action should implement :meth:`activate() <depthai_sdk.trigger_action.Action.activate>` and optionally :meth:`on_new_packets() <depthai_sdk.trigger_action.Action.on_new_packets>` methods.

The set of predefined actions:

- :class:`RecordAction <depthai_sdk.trigger_action.RecordAction>` - records a video of a given duration when a trigger is activated.

Usage
-----

The following example shows how to create a trigger that is activated when at least 1 person is detected in the frame.
When the trigger is activated, it records a 15 seconds video (5 seconds before the trigger is activated and 10 seconds after).

.. literalinclude:: ../../../examples/trigger_action/person_record.py
   :language: python

Reference
---------

.. autoclass:: depthai_sdk.trigger_action.TriggerAction
    :members:
    :undoc-members:

.. autoclass:: depthai_sdk.trigger_action.Trigger
    :members:
    :undoc-members:

.. autoclass:: depthai_sdk.trigger_action.Action
    :members:
    :undoc-members:

.. autoclass:: depthai_sdk.trigger_action.DetectionTrigger
    :members:
    :undoc-members:

.. autoclass:: depthai_sdk.trigger_action.RecordAction
    :members:
    :undoc-members:

