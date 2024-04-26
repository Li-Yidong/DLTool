# GUIと.jsonファイルについて

このファイルでは、`GUI`の作成に使用される`json/ui.json`ファイルの変更方法を説明する。定義されている`UI`要素のタイプ、`GUI`のどの部分が静的か動的か、コードにフィーチャーを追加する方法について説明する。

`.json`ファイルは生成されるコンフィグファイルの構造も定義する。生成される設定ファイルの構造は同じですが、辞書のすべての部分に `pyqt`キーがない。

最初にエレメントをレンダリングするときの`GUI`は以下のようになる：

![Overall UI](./gui/app-overall.png)

## 静的要素説明

静的な`UI`要素は、.jsonファイルの構造に関係なくレンダリングされます。

以下は静的な`UI`要素である：
1. [言語](#language-option)
2. [view](#view-option)
3. [save](#save-option)
4. [load](#load-option)
5. [run](#run-option)

![Static Elements](./gui/element-static.png)

### 言語
`言語`ボタンは、UIを英語または日本語に翻訳するために使用する。

このボタンは切り替え可能で、現在どの言語が使用されているかを示す。

### View

`view`ボタンは、.jsonまたは辞書の生のフォーマットで、コンフィグファイルを表示するために使用されます。

viewボタンを使用した例以下のようになる：

![View Option](./gui/page-view)

### Save

`Save`ボタンは、コンフィグファイルを.json形式で保存し、`Load`で読み込むことができます。

### Load

`Load`ボタンは、コンフィグした設定ファイル（.json形式）をロードし、それに応じて`UI`を更新します。

### Run

`run`ボタンを押すと今のコンフィグレーションで学習/テスト/プルーニングを実行する

`run`ボタンを押すと`GUI`が自動で最小化される。

## 動的要素説明

### 概要

`.json`ファイルで定義された全ての要素は、`GUI`で`UI`要素を定義するために使用される "pyqt "キーを持っています。

動的UI要素タイプセクションのすべてのサブセクションは、[この](#overview)形式で説明します：

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