## Accessible PDF Demo Document

This document has been created to demo tagging lists, artifacts, tables, links, and images, and is not meant to address all the requirements needed for a PDF to be compliant with WCAG 2.1 or ISO 142891 (PDF/UA -1). Please independently review or seek independent counsel to review and identify the relevant requirements of accessibility standards that may apply in your use case.

## Lists

When crea�ng lists, the list style in Word should be used. Below is a sample simple list.

## Simple List

List of Canadian Provinces by Popula�on (2014)

1. Ontario
2. Quebec
3. Bri�sh Columbia
4. Alberta
5. Manitoba
6. Saskatchewan
7. Nova Sco�a
8. New Brunswick
9. Newfoundland and Labrador Prince

## Complex List

For our purposes a complex list is one with sub -bullets or items. This example also includes mul�ple paragraphs within a single bullet. The tag tree will be much deeper for this type of list. This would be true whether it is a numbered list or a bulleted list. For example:

## Some Adobe Crea�ve Cloud Desktop Apps with Descrip�ons

- Adobe Acrobat is a so�ware family dedicated to Adobe's Portable Document Format (PDF). There are two releases of Adobe Acrobat that support the crea�on of PDF files:
- o Acrobat Pro DC Classic release
- o Acrobat Pro DC Con�nuous

Adobe Reader is an applica�on that allows the reading of PDF files.

- Adobe A�er Effects is a digital mo�on graphics and composi�ng so�ware published by Adobe Systems. It is o�en used in film and video post -produc�on.
- Adobe Animate is a vector anima�on so�ware used to design interac�ve anima�ons with drawing tools to publish them on mul�ple pla�orms like Adobe Flash, Adobe AIR, HTML5 canvas, WebGL.
- It is the successor to Adobe Flash Professional and also includes features of Adobe Edge, which is discon�nued.
- o Adobe Flash Builder, formerly Adobe Flex Builder, is an integrated development environment (IDE) built on the Eclipse pla�orm meant for developing rich Internet applica�ons (RIAs) and cross -pla�orm desktop applica�ons for the Adobe Flash pla�orm.
- o Adobe Scout, a profiling tool for Flash SWF files.
- Adobe Bridge is an organiza�onal program.
- Its primary purpose is to link the parts of the Crea�ve Suite together using a format similar to the file browser found in previous versions of Adobe Photoshop.
- Adobe Photoshop is a raster -graphics editor (with significant vector graphics func�onality).
- o Adobe Photoshop Lightroom is a photo processor and image organizer.
- Adobe Premiere Pro is a real -�me, �meline -based video edi�ng so�ware applica�on. Its relate d applica�ons are:
- o Adobe Media Encoder, a tool to output video files.
- o Adobe Prelude, a tool for impor�ng (inges�ng), reviewing, and logging tapeless media.
- o Adobe SpeedGrade, a tool for performing color correc�ons and developing looks for Premiere proũects.

## Ar�facts

According to the PDFͬhA ISO standard, content that does not represent meaningful content, or appears as a background, shall be tagged as an ar�fact. Examples of such content include decora�ve images or line spaces.

Addi�onally, because they are not considered real content, ar�facts are not present in the structure tree (or in Adobe Acrobat 's Tag Tree).

One common error in a Word document is to use blank lines to create space between paragraphs instead of using paragraph spacing. This may create the desired visual effect but, unless the blank line is tagged as an ar�fact, it may be read by the screen reader as ' blank line͟.

Another element that the author may decide to tag as an ar�fact is an image that has a cap�on. Par�cularly if the cap�on contains all the relevant informa�on about the image in context in the document. Adding alt text to this image would create redundant informa�on for the screen reader and not tagging it as an ar�fact may confuse the user as the screen reader will indicate ' image ' but with not associated alt text.

<!-- image -->

Figure 1: My dog with his goofy grin

<!-- image -->

Figure 2: A new neighbor comes to visit

## Tables

It is possible when authoring tables in Word to indicate the column and row headers for simple tables. Depending upon the method of conversion the correct &lt;TH&gt; tag (table header) may be generated. However there is likely manual clean up to do for table accessibility, par�cularly for more complex tables.

## Simple Table

Vehicles Sold by Model and Color

| Color | SUVs | Sedans | Trucks |
|----------------|--------|----------|----------|
| Ruby Red | 4 | 4 | 5 |
| Midnight Black | 7 | 5 | 5 |
| Triple Yellow | 3 | 5 | 7 |

## Complex Table

Sales Results for 2015

Images

| Salesperson | 1 st Quarter 2015 | 1 st Quarter 2015 | 1 st Quarter 2015 | 2 nd Quarter 2015 3 | 2 nd Quarter 2015 3 | 2 nd Quarter 2015 3 | rd Quarter 2015 | rd Quarter 2015 | rd Quarter 2015 | 4 th Quarter 2015 | 4 th Quarter 2015 | 4 th Quarter 2015 | Annual Total |
|---------------|---------------------|---------------------|---------------------|-----------------------|-----------------------|-----------------------|-------------------|-------------------|-------------------|---------------------|---------------------|---------------------|----------------|
| | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec | |
| Susan | 135 | 250 | 235 | 200 | 250 | 235 | 245 | 175 | 225 | 250 | 175 | 125 | 2500 |
| Bill | 175 | 275 | 250 | 225 | 235 | 210 | 200 | 150 | 200 | 275 | 250 | 175 | 2620 |
|:eff | 100 | 150 | 175 | 200 | 225 | 250 | 200 | 175 | 200 | 250 | 200 | 150 | 2275 |

## Photos of Sunsets

<!-- image -->

V

&lt;Link&gt;

&lt; Link&gt;

Adobe website

Link - OBJR

Link - OBJR

## Links

There are three ways in which links are indicated within a document. Using the full URL, using an abbreviated URL, and embedding the link the text of the document (preferred). Each approach presents unique challenges in making a PDF accessible to the screen reader user. Embedding the link in the text of the document is the best approach as it provides the most readable and understandable form of the link. The following sentences demonstrate three ways to add a hyperlink to a document:

## Embedded Hyperlink

Check out our Adobe accessibility website for more informa�on about accessibility at Adobe.

## Shortened URL

For more informa�on on accessibility at Adobe click on the link,.

## Full URL

Go to hƩp:ͬͬwww.adobe.comͬaccessibility for more informa�on Adobe accessibility.

A correctly tagged link has the &lt;Link&gt; tag, text associated with the link, and a Link - OBJR tag (the order of the last two is irrelevant.

Figure 1: Link tagged in the order of &lt;Link&gt; tag, link text, and Link - OBJR tag

<!-- image -->

Figure 2:: Link tagged in the order of &lt;Link&gt; tag, Link - OBJR tag, and link text

<!-- image -->