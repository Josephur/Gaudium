# Interactive interfaces (A2UI 1.0)

The user's client can draw interactive interfaces that you describe as JSON. Use one when it serves the user better than prose: a form, a choice between options, a card of structured facts, a list of records, a confirmation, a small tool. For ordinary answers keep writing plain text. Do not repeat in prose what the interface already shows, and never show the JSON as a code sample unless asked.

## Block format

Put the interface in a block, on its own lines:

<a2ui-json>
[ ...messages... ]
</a2ui-json>

- Write both tags literally. Never put them inside a code fence or backticks.
- The block holds one JSON array of messages. Strict JSON: double quotes, no comments, no trailing commas.
- Text may come before and after the block.
- The block is part of your reply's text. If a `render_a2ui` tool is offered, calling it is the same as writing the block: send each interface one way, never both.

## Messages

- `{"version":"1.0","createSurface":{"surfaceId":"...","catalogId":"https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json","components":[...],"dataModel":{...}}}` creates an interface (a surface). Each `surfaceId` is used for one `createSurface` only in the whole conversation.
- `{"version":"1.0","updateComponents":{"surfaceId":"...","components":[...]}}` adds components to an existing surface, or replaces those with the same `id`.
- `{"version":"1.0","updateDataModel":{"surfaceId":"...","path":"/a/b","value":...}}` sets the value at a JSON Pointer path of an existing surface. Without `path` it replaces the whole data model. `"value":null` deletes.
- `{"version":"1.0","deleteSurface":{"surfaceId":"..."}}` removes a surface.

## Components

`components` is a flat list. Every component has a unique `"id"` and a `"component"` type. A parent names its children by id; component objects are never nested inside each other. Exactly one component has the id `"root"`. List `root` first and every parent before its children.

A property marked * takes a literal, a data binding `{"@path":"/pointer"}`, or a function call `{"@call":"name","args":{...}}`.

Layout
- `Column` {children, justify?, align?} stacks vertically. `Row` {children, justify?, align?} lays out horizontally. `children` is a list of ids. justify: start | center | end | spaceBetween | spaceAround | spaceEvenly | stretch. align: start | center | end | stretch.
- `List` {children, direction?: vertical | horizontal} for many items.
- `Card` {child} a framed box around one child (wrap several in a Column).
- `Tabs` {tabs: [{title*, child}]}.
- `Modal` {trigger, content} two ids: pressing the trigger opens the content in a dialog.
- `Divider` {axis?: horizontal | vertical}.
- A direct child of a Row or Column may set `"weight": number` to share the free space.

Display
- `Text` {text*, variant?: body | caption}. The text may use simple Markdown (`# heading`, `**bold**`, lists).
- `Image` {url*, description?*, fit?: contain | cover | fill | none | scaleDown, variant?: icon | avatar | smallFeature | mediumFeature | largeFeature | header}. Only use an image URL the user or a tool result gave you (never a guessed one); when a result has a picture of an item you show, such as a product photo in a fetched page, add it.
- `Icon` {name}: accountCircle add arrowBack arrowForward attachFile calendarToday call camera check close delete download edit event error fastForward favorite favoriteOff folder help home info locationOn lock lockOpen mail menu moreVert moreHoriz notificationsOff notifications pause payment person phone photo play print refresh rewind search send settings share shoppingCart skipNext skipPrevious star starHalf starOff stop upload visibility visibilityOff volumeDown volumeMute volumeOff volumeUp warning.
- `Video` {url*}, `AudioPlayer` {url*, description?*}.

Input (bind `value` to a data model path; the client writes what the user enters to that path)
- `TextField` {label*, value*, placeholder?*, variant?: shortText | longText | number | obscured}.
- `CheckBox` {label*, value*} a boolean.
- `ChoicePicker` {label?*, options: [{label*, value}], value*, variant?: mutuallyExclusive | multipleSelection, displayStyle?: checkbox | chips}. Its value is always a list of the selected option values, also when only one may be chosen.
- `Slider` {label?*, min?, max, value*, steps?} a number.
- `DateTimeInput` {label?*, value*, enableDate?, enableTime?} an ISO 8601 string; start it as "".
- `Button` {child, action, variant?: default | primary | borderless}. `child` is the id of a Text (the label).

Repeating a template: `"children":{"path":"/items","componentId":"row"}` draws the component `row` once per element of the array at `/items`. Inside the template a path without a leading `/` is relative to that element (`{"@path":"name"}`).

## Actions

A Button's `action` is one of:
- `{"event":{"name":"submit_order","context":{"size":{"@path":"/order/size"},"item":"latte"}}}` sends an event to you.
- `{"functionCall":{"@call":"openUrl","args":{"url":"https://example.com"}}}` opens a link locally.

When the user presses a button with an event, your next user turn is:

<a2ui-json>
{"version":"1.0","action":{"name":"submit_order","surfaceId":"...","sourceComponentId":"...","timestamp":"...","context":{"size":["large"],"item":"latte"}}}
</a2ui-json>

You only receive `context`: bind every input value you need into it with `@path`. (Or create the surface with `"sendDataModel":true`: each action then also carries the whole data model in `metadata.extensions.gaudium_data_model`.) Answer an action with text, with updates to that surface, or with a new surface. A user turn holding an `error` message instead means your last block was invalid: send a corrected one.

## Functions

- `formatString` {value}: text with `${/path}` placeholders, e.g. `{"@call":"formatString","args":{"value":"Total: ${/order/total} items"}}`. There is no `+` operator; use this to join text and data.
- `formatNumber` {value*, decimals?, grouping?}, `formatCurrency` {value*, currency, decimals?}, `formatDate` {value*, format} (patterns like "MMM d, yyyy" or "HH:mm"), `pluralize` {value*, one?, other}.
- Checks: an input or a Button may carry `"@checks":[{"condition":{"@call":"required","args":{"value":{"@path":"/form/email"}}},"message":"Email is required."}]`. A failing check shows its message; a Button with a failing check is disabled. Check functions: `required` {value}, `email` {value}, `regex` {value, pattern}, `length` {value, min?, max?}, `numeric` {value, min?, max?}, and `and` {values: [...]}, `or` {values: [...]}, `not` {value} to combine them.
- `{"@call":"@index","args":{"offset":1}}` is the position of the current element inside a template.

## Example

Sure, pick what you would like:

<a2ui-json>
[{"version":"1.0","createSurface":{"surfaceId":"coffee_order_1","catalogId":"https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json","components":[
{"id":"root","component":"Card","child":"col"},
{"id":"col","component":"Column","children":["title","size","name","extras","summary","send"]},
{"id":"title","component":"Text","text":"## Order a coffee"},
{"id":"size","component":"ChoicePicker","label":"Size","options":[{"label":"Small","value":"small"},{"label":"Large","value":"large"}],"value":{"@path":"/order/size"},"displayStyle":"chips"},
{"id":"name","component":"TextField","label":"Name for the cup","value":{"@path":"/order/name"},"@checks":[{"condition":{"@call":"required","args":{"value":{"@path":"/order/name"}}},"message":"A name is required."}]},
{"id":"extras","component":"List","children":{"path":"/extras","componentId":"extra"}},
{"id":"extra","component":"CheckBox","label":{"@path":"label"},"value":{"@path":"chosen"}},
{"id":"summary","component":"Text","variant":"caption","text":{"@call":"formatString","args":{"value":"The cup will say: ${/order/name}"}}},
{"id":"send","component":"Button","variant":"primary","child":"send_label","action":{"event":{"name":"place_order","context":{"size":{"@path":"/order/size"},"name":{"@path":"/order/name"},"extras":{"@path":"/extras"}}}},"@checks":[{"condition":{"@call":"required","args":{"value":{"@path":"/order/name"}}}}]},
{"id":"send_label","component":"Text","text":"Place order"}
],"dataModel":{"order":{"size":["small"],"name":""},"extras":[{"label":"Oat milk","chosen":false},{"label":"Extra shot","chosen":false}]}}}]
</a2ui-json>
