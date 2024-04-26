# GUI and JSON file explained

This readme file explains how to modify the json/ui.json file that is used to create the GUI. This documentation explains the types of UI element that is defined, state which parts of the GUI are static or dynamic, and how to add more types to the code.

The .json file also defines the structure of the produced configuration file. The structure of the resulting config file is the same, just without the "pyqt" key in every part of the dictionary.

Below is an example of how the GUI would look like when first rendering the elements.

![Overall UI](./gui/app-overall.png)

## Static UI Element

Static UI elements are rendered regardless of the .json file structure.

Below are the static UI elements that are rendered (in order from left to right: [language option](#language-option), [view option](#view-option), [save option](#save-option), [load option](#load-option), and [run option](#run-option))

![Static Elements](./gui/element-static.png)

### Language Option

The language button is used to translate the UI to English or Japanese.

The button is a toggleable button and indicates which language is currently used.

### View Option

The view button can be used to view the would-be or resulting configuration file in its raw .json or dictionary format.

Below is an example on using the view option.

![View Option](./gui/page-view)

### Save Option

The save button saves the configuration file in the format of .json that can be loaded for the following sessions.

The save button triggers a Windows file dialog UI to save the configuration file.

### Load Option

The load button will load a chosen configuration file (.json format) and updates the UI accordingly.

The load button triggers a Windows file dialog UI to load the configuration file.

### Run Option

The run button is used to run the main python file.

The run button also triggers the window to be **minimized**, because the GUI would be unresponsive when the main python file is run.

## Dynamic UI Element Types

### Overview

Every element defined in the .json file has a "pyqt" key used to define the UI element in the GUI.
Every subsection in the [Dynamic UI Element Types](#overview) section will have this format for explanation:

1. What the UI element looks like in the GUI

2. How the UI element is defined in the .json file (specifically the "type", "name", and other required keys for the UI element that has to be in its "pyqt" key)

*Every UI element defined will have a "comment" key that will work the same for everything. The "comment" key provides comments that will show up as a tooltip when a mouse hovers over the UI element. So, the "comment" key will not be explained or shown in every subsection. A tooltip comment would look something like below.

![Tooltip Comment](./gui/element-tooltip.png)

*Every "name", "comment" keys are a list of string for translation purposes, the keys and values shown in every other subsection is a simplified version of what would be actually written to the ui.json file

Below is an example of how a UI element would be defined in the .json file.

```json
"model-selection": {
    "pyqt": {
        "type": "Dropdown",
        "name": [
            "Model Architecture",
            "モデルアーキテクチャ"
        ],
        "comment": [
            "Model Architecture selection",
            "モデルアーキテクチャの選択"
        ],
        "choices": [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152"
        ],
        "default": "resnet18"
    }
}
```

### Menu / Section Type

```json
"type": "Dict"
"name": "Button Name"
```

The menu element is different from other elements. The way the menu element is rendered is different per level of the dictionary. "Level" referenced in this explanation means how deep is it in the dictionary file, for example:

The "name" key would be shown as the name of the section or would be shown on the navigation button if it is a level 0 element.

```json
"level0": {
    "level1": {
        "level2": "value"
    }
}
```

A level 0 Dict element would be rendered as a button menu in the sidebar of the GUI as seen below.

Levels 1 to 2 Dict element would be rendered with varying bold levels.

*The code provides the option up until level 3 of Dict element, but the configuration file right now only define elements up until level 2.

![Dict Level 0](./gui/element-dict-level-0.png)

![Dict Level 1](./gui/element-dict-level-1.png)

![Dict Level 2](./gui/element-dict-level-2.png)

### Checkbox Input

```json
"type": "Boolean"
"name": "Button Name"
"default": true | false
```

The "name" key sets the text that will show up besides the checkbox in the UI element.

The "default" key sets whether the Checkbox is initially checked or unchecked.

The UI element looks something like below.

![Checkbox Element](./gui/element-checkbox.png)

### User Keyboard Input

```json
"type": "StrInput" | "IntInput" | "FltInput" | "ShpInput" | "SizeInput" | "ListInput"
"name": "Button Name"
"default": "Default Value (str, int, float, list, etc.)"
```

The "name" key sets the text that will show up behind the keyboard input element inside the UI element.

The "default" key is the default value with the type corresponding to the type of keyboard input defined that would show up when first rendering the GUI. The default value for StrInput is of type str, for IntInput is of type integer, and so on.

The input is validated by a regex, so the user should not be able to input a wrong value type into the text field.

The UI elements look something like below. (In order from top to bottom: StrInput, IntInput, FltInput, ShpInput, SizeInput)

![String Input](./gui/element-input-str.png)

![Integer Input](./gui/element-input-int.png)

![Float Input](./gui/element-input-float.png)

![Shape Input](./gui/element-input-shape.png)

![Size Input](./gui/element-input-size.png)

![List Input](./gui/element-input-list.png)

### Dropdown Selection

```json
"type": "Dropdown"
"name": "Button Name"
"choices": ["Selection 1", "Selection 2"]
"default": "Selection 1"
```

The "name" key sets the text that will show up besides the dropdown box in the UI element.

The "default" key is the default choice selected when first rendering the checkbox UI element.

The "choices" key contains choices that can be selected by the user.

The UI element looks something like below.

![Dropdown](./gui/element-dropdown.png)

### Path Selection

```json
"type": "Path"
"name": "Button Name"
"target": "Folder" | "File"
"default": "Default/path/to/folder/or/file"
```

The "name" key sets the text that will show up on the button that starts the file selection dialog.

The "default" key is the default path and will be shown when first rendering the Path Selection element besides the button.

The "target" key determines the target of the path selection, whether to choose a folder or a file. (The value must be either Folder or File)

![Path Selection](./gui/element-path.png)