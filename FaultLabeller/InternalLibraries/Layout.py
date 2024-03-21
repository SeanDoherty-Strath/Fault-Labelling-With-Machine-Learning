import InternalLibraries.Components as Components
from dash import html
import InternalLibraries.Styles as Styles

mainLayout = html.Div(style=Styles.window, children=[

    # MODALS
    Components.commentModal,
    Components.alertOneContainer,
    Components.alertTwoContainer,

    # CONTENTS
    Components.title,
    Components.topBox,

    # BOTTOM BOXES
    html.Div(style=Styles.bottomBoxes, children=[
        # Box 1: Variables to Plot
        html.Div(style=Styles.Box1,
                 children=[
                     Components.sensorHeader,
                     Components.sensorText,
                     Components.sensorDropdown,
                     Components.xAxis,
                     Components.yAxis,
                     Components.zAxis

                 ]),

        html.Div(style=Styles.column, children=[
            #  Box 2: Manual Label
            html.Div(style=Styles.Box2, children=[
                html.Div(style=Styles.Box2Container, children=[
                    Components.labelTitle,
                    Components.labelDropdown,
                    html.Button(id='labelButton', style=Styles.labelButton),
                    html.Button('Remove Labels', id='removeLabels',
                                style=Styles.removeLabels),
                ])]),

            html.Div(style=Styles.emptySpace),

            #  Box 3: Navigation
            html.Div(style=Styles.Box3, children=[
                # html.Div(style=Styles.Box3Container, children=[
                Components.navigatorTitle,
                html.Div(style=Styles.Box3Container2, children=[
                    Components.navigationText,
                    Components.faultFinder,
                ]),
                Components.navigationButtons
            ])
        ]),

        # Box 4: Auto Labelling
        html.Div(style=Styles.Box4, children=[
            Components.AI_header,
            Components.AI_text1,
            Components.AI_text2,
            Components.AI_text3,
            Components.AI_selectButtons,
            Components.AI_sensorChecklist,
            Components.AI_text4,
            Components.AI_text5,

            html.Div(style=Styles.flex, children=[
                Components.AI_text6,
                Components.reductionMethod
            ]),
            html.Div(style=Styles.flex, children=[
                Components.AI_text7,
                Components.reducedSize

            ]),
            Components.uploadNewAutoencoder,
            Components.AI_text8,
            Components.AI_text9,

            html.Div(style=Styles.flex, children=[
                Components.AI_text10,
                Components.clusterMethod,
            ]),

            html.Div(style=Styles.flex, children=[
                Components.AI_text11,
                Components.K
            ]),

            Components.uploadTrainingData,
            html.Div(id='epsilon', style=Styles.sliderContainer, children=[
                Components.AI_text12,
                html.Div(style=Styles.slider, children=[
                    Components.epsSlider
                ])
            ]),
            html.Div(id='minVal', style=Styles.sliderContainer, children=[
                Components.AI_text13,
                html.Div(style=Styles.slider, children=[
                    Components.minPtsSlider
                ]),
            ]),
            html.Button(children='Start Clustering',
                        id='startAutoLabel', style=Styles.startButton)
        ]),

        #    Box 5: Stats
        html.Div(style=Styles.Box5, children=[
            html.Div(style=Styles.Box5Container,  children=[
                Components.Box5text,
                Components.stat1, Components.stat2, Components.stat3,
            ]
            ),
            html.Div(style=Styles.emptySpace),

            # Box 6: Import and export
            html.Div(style=Styles.Box6, children=[
                html.Div(style=Styles.Box6Container, children=[
                    Components.Box6text,
                    html.Div(style=Styles.Box6Container2, children=[
                        Components.exportName,
                        Components.includeCommentsButton,
                        Components.exportConfirm]),
                    Components.downloadData,
                    Components.uploadData,
                ])
            ])
        ]),
    ]),
])
