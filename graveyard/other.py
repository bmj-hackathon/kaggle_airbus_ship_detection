

if 0: # DEV
    image_id = df_by_image.index[2] # Select an image with 15 ships
    image_id = df_by_image.index[-2]
    selfimage = Image(image_id)
    selfimage.records
    selfimage.load(img_zip, df)
    selfimage.load_ships()
    df_summary = selfimage.ship_summary_table()
    df_summary
    # for idx,  in selfimage.records.iterrows():
    #     print(i['contour'])

    # i
    # selfimage.records['contour']

    canvas2 = selfimage.draw_ellipses_img()

    # for i in
    #     print(i )

    # r = image.records

